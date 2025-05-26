import copy
from typing import List
import scipy.optimize as opt
import scipy.io as io
import time
import calendar as cal
import datetime
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np

from pymgipsim.Controllers.SMDI.Estimation.DEIVP import DifferentialEvolution
from pymgipsim.Utilities.Scenario import scenario, inputs
from pymgipsim.InputGeneration.signal import Signal, Events
from scipy.stats import qmc
from copy import deepcopy
from pymgipsim.ModelSolver.BaseSolvers import BaseSolver
from pymgipsim.VirtualPatient.Models.T1DM import IVP
from pymgipsim.InputGeneration.carb_energy_settings import generate_carb_absorption
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Estimator:
    """ Observer class.

    """

    def __init__(self, scenario: scenario, patient_idx):
        """ Constructor.

        """
        self.lower_bounds: np.ndarray = np.array([], dtype=np.float32)
        self.upper_bounds: np.ndarray = np.array([], dtype=np.float32)
        self.interpolated_glucose_level: np.ndarray = np.array([], dtype=np.float32)
        self.optimized_params: List[float] = []
        self.steady_state_glucose_lower_bound = 80.0
        self.steady_state_glucose_upper_bound = 130.0
        self.basal_insulin: float = UnitConversion.insulin.Uhr_to_uUmin(scenario.patient.demographic_info.basal[0])
        self.with_meal_estimation = False
        self.number_of_meal_param_coeff = 1
        self.max_iterations = 1000
        self.optimization_random_seed = 1
        self.optimization_threads = np.inf
        self.is_historical_init_state = False
        self.population_size = 150
        self.verbose = False
        self.is_multiple_taud = True
        self.is_interval_estimation = True
        self.no_meals = 0
        self.no_sections = 1
        self.optimizer = DifferentialEvolution()
        self.sim_sampling_time = scenario.settings.sampling_time
        self.scenario = deepcopy(scenario)
        if self.with_meal_estimation:
            self.number_of_meal_param_coeff = 2
        self.scenario.patient.model.name = IVP.Model.name
        self.scenario.patient.number_of_subjects = 1
        self.scenario.settings.end_time = 5*96
        self.scenario.settings.sampling_time = 5
        self.scenario.settings.solver_name = "Euler"
        self.scenario.patient.demographic_info.body_weight = [self.scenario.patient.demographic_info.body_weight[0]]
        self.patient_parameters = IVP.Parameters(IVP.Parameters.generate(self.scenario))


    def run(self, sample, patient_idx, measurements, insulins):
        """ Implements parameter estimation based on the data provided in simulation_data.

        Args:
            simulation_data:

        Returns:

        """
        self.interpolated_glucose_level = np.asarray(measurements)

        self.insulins = np.asarray(insulins)
        self.scenario.settings.end_time = int(sample*self.sim_sampling_time)
        self.time = np.arange(0,self.scenario.settings.end_time+self.scenario.settings.sampling_time,self.scenario.settings.sampling_time)
        binmap = np.asarray(self.scenario.inputs.meal_carb.start_time[patient_idx])<(sample-self.scenario.settings.sampling_time)
        self.meals = np.asarray(self.scenario.inputs.meal_carb.magnitude[patient_idx])[binmap]
        self.meal_times = np.asarray(self.scenario.inputs.meal_carb.start_time[patient_idx])[binmap]
        self.no_meals = np.sum(binmap)

        # Lower bounds for the optimization.
        self.lower_bounds = np.array([IVP.CONSTANTS.MIN_TAU1,  # tau1
                                      IVP.CONSTANTS.MIN_TAU2,  # tau2
                                      15.0,  # p2
                                      IVP.CONSTANTS.MIN_VG])  # Vg
        # Upper bounds for the optimization.
        self.upper_bounds = np.array([IVP.CONSTANTS.MAX_TAU1*2,  # tau1
                                      IVP.CONSTANTS.MAX_TAU2*2,  # tau2
                                      100.0*2,  # p2
                                      IVP.CONSTANTS.MAX_VG])  # Vg

        # adds as many time constants as many meals are taken during the current identification horizon
        self.lower_bounds = np.concatenate((self.lower_bounds, [15.0]*self.no_meals))
        self.upper_bounds = np.concatenate((self.upper_bounds, [80.0] * self.no_meals))
        self.lower_bounds = np.concatenate((self.lower_bounds, [IVP.CONSTANTS.MIN_EGP, IVP.CONSTANTS.MIN_GEZI, IVP.CONSTANTS.MIN_SI]))
        self.upper_bounds = np.concatenate((self.upper_bounds, [IVP.CONSTANTS.MAX_EGP, IVP.CONSTANTS.MAX_GEZI, IVP.CONSTANTS.MAX_SI]))

        # Create initial parameter population with Latin Hypercube Sampling.
        sampler = qmc.LatinHypercube(len(self.lower_bounds),seed=np.random.default_rng(42))
        self.parameter_array_init = np.add(np.multiply(sampler.random(self.population_size),
                                    np.expand_dims(self.upper_bounds - self.lower_bounds, 0)),
                                         np.expand_dims(self.lower_bounds, 0))

        # Run the diffenertial evolutation optimizer to get the patient parameters
        self.optimized_params = self.optimizer.run(self)

        if(self.verbose):
            parameter_VA = self.optimized_params
            print(np.asarray(parameter_VA[4:4 + self.no_meals])-self.lower_bounds[4])
            print("Parameter estimation results------------------------------")
            print("tau1:", "{:.2f}".format((parameter_VA[0] - self.lower_bounds[0]) / (self.upper_bounds[0]-self.lower_bounds[1])), end="  ")
            print("tau2:", "{:.2f}".format((parameter_VA[1] - self.lower_bounds[1]) / (self.upper_bounds[1]-self.lower_bounds[1])), end="  ")
            print("p2:", "{:.2f}".format((parameter_VA[2] - self.lower_bounds[2]) / (self.upper_bounds[2]-self.lower_bounds[2])), end="  " )
            print("Vg:", "{:.2f}".format((parameter_VA[3] - self.lower_bounds[3]) / (self.upper_bounds[3]-self.lower_bounds[3])))
            idx = 4 + self.number_of_meal_param_coeff * self.no_meals
            EGPn = np.divide(np.subtract(np.asarray(parameter_VA[idx::3]),self.lower_bounds[idx]),self.upper_bounds[idx]-self.lower_bounds[idx])
            print("EGP:", *list(map(lambda x:"{:.2f}".format(x),EGPn)))
            idx = 4 + self.number_of_meal_param_coeff * self.no_meals+1
            GEZIn = np.divide(np.subtract(np.asarray(parameter_VA[idx::3]),self.lower_bounds[idx]),self.upper_bounds[idx]-self.lower_bounds[idx])
            print("GEZI:", *list(map(lambda x:"{:.2f}".format(x),GEZIn)))
            idx = 4 + self.number_of_meal_param_coeff * self.no_meals+2
            SIn = np.divide(np.subtract(np.asarray(parameter_VA[idx::3]),self.lower_bounds[idx]),self.upper_bounds[idx]-self.lower_bounds[idx])
            print("SI:", *list(map(lambda x:"{:.2f}".format(x),SIn)))
            #print("taud:", "{:.2f}".format((np.asarray(parameter_VA[4:4 + self.no_meals])-self.lower_bounds[4])/(self.upper_bounds[4]-self.lower_bounds[4])))
            taudn = (np.asarray(parameter_VA[4:4 + self.no_meals])-self.lower_bounds[4])/(self.upper_bounds[4]-self.lower_bounds[4])
            print("taud:", *list(map(lambda x: "{:.2f}".format(x), taudn)))
            print("---------------------------------------------------------")

        self.cost_function(self.optimized_params)

        # Fitted patient parameters are in:
        # self.scenario.patient.model.parameters
        # Last state:
        # self.solver.model.states.as_array[0, :, -1]
        self.avg_carb_time = np.asarray(self.optimized_params[4:4 + self.no_meals]).mean()
        # plt.figure()
        # plt.plot(measurements)
        # plt.plot(self.solver.model.states.as_array[0,0,:])
        # plt.legend(["CGM","Base solver with fitted params"])
        # plt.show()

    def cost_function(self, optimization_params: List[float]):
        """ Cost function of the identification.

        """
        # patient_parameters = self.scenario.patient.model.parameters
        self.scenario.patient.model.parameters = [[self.scenario.patient.demographic_info.body_weight[0],
                                 optimization_params[4 + self.number_of_meal_param_coeff * self.no_meals],
                                 optimization_params[4 + self.number_of_meal_param_coeff * self.no_meals+1],
                                 optimization_params[4 + self.number_of_meal_param_coeff * self.no_meals+2],
                                 IVP.CONSTANTS.NOMINAL_CI,
                                 optimization_params[0],
                                 optimization_params[1],
                                 1.0/optimization_params[2],
                                 optimization_params[3]]]

        ivp_inputs = inputs()
        ivp_inputs.meal_carb = Events([self.meals],[self.meal_times],[np.ones_like(self.meal_times)])
        ivp_inputs.taud = Events([np.asarray(optimization_params[4:4 + self.no_meals])],[self.meal_times],[np.ones_like(self.meal_times)])
        ivp_inputs.basal_insulin = Events([[0.0]],[[0.0]])
        ivp_inputs.bolus_insulin = Events([[0.0]],[[0.0]])
        hor_scenario = copy.deepcopy(self.scenario)
        hor_scenario.inputs = ivp_inputs

        model = IVP.Model.from_scenario(hor_scenario)
        # print(self.insulins.shape)
        model.inputs.basal_insulin.sampled_signal = self.insulins[None,:]

        self.solver = BaseSolver(hor_scenario, model)
        self.solver.model.preprocessing()

        self.solver.do_simulation(True)
        error_diff = np.subtract(self.interpolated_glucose_level, self.solver.model.states.as_array[0,0,:])

        # RMSE of the identification
        J = np.sqrt(np.mean(error_diff ** 2))

        # print(J)
        # print(self.solver.model.states.as_array[0,0,:])
        return J