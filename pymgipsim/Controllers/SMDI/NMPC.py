import numpy as np
import timeit
import os

from fontTools.misc.cython import returns
from matplotlib import pyplot as plt

from pymgipsim.InputGeneration.signal import Events
from pymgipsim.Utilities.Scenario import scenario
from copy import deepcopy
from pymgipsim.ModelSolver.BaseSolvers import BaseSolver
from pymgipsim.VirtualPatient.Models.T1DM import IVP
from pymgipsim.InputGeneration.carb_energy_settings import generate_carb_absorption
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class NMPC:
    """ Impulsive model predictive controller class with saturation.

    """

    def __init__(self, scenario: scenario, patient_idx: int):

        self.patient_idx = 0
        self.ctrl_sampling_time = 5
        self.steps = int((scenario.settings.end_time - scenario.settings.start_time) / scenario.settings.sampling_time)
        self.ideal_glucose = 110
        self.glucose_target = np.zeros(self.steps)
        # Create containers
        self.insulin = np.zeros(self.steps)
        self.positive_insulin = np.zeros(self.steps)
        self.cgms = np.zeros(self.steps)

        self.multiple_taud = False
        self.verbose = False
        self.count_meals = 0
        self.iterator = 0
        self.total_cost = 0
        self.counter_array = np.zeros((self.steps,))
        self.control_horizon = 1
        self.prediction_horizon = 120
        self.gradient_delta = 100.0
        self.max_grad_counter = 500
        self.epsilon = 1e-6
        self.grad_gamma = 0.9
        self.grad_stepsize = 10.0
        self.insulin_limit = 25e6  # hard constraint, 25 U
        self.last_min_diff = np.zeros(self.steps)
        self.is_multiple_taud = True

        self.scenario = deepcopy(scenario)
        self.scenario.patient.model.name = IVP.Model.name
        self.scenario.patient.number_of_subjects = 1
        self.scenario.patient.demographic_info.body_weight = [self.scenario.patient.demographic_info.body_weight[patient_idx]]
        self.scenario.patient.model.parameters = IVP.Parameters.generate(self.scenario)
        self.scenario.inputs.taud = generate_carb_absorption(self.scenario,None)
        self.basal_rate = self.scenario.patient.demographic_info.basal[patient_idx]
        # self.model = IVP.Model.from_scenario(self.ctrl_scenario)

        # Initialize arrays
        # self.approx_x_init = self.solver.model.get_basal_equilibrium(self.solver.model.parameters.as_array, self.ideal_glucose)
        # self.approx_x_current = self.approx_x_init
        # self.insulin_init = 1200.0 * self.approx_x_init # self.patient_params.CI * 
        # print("Insulin init: ", self.insulin_init)

        # Hardcoded init values for testing before identification algoritm
        self.glucose_init = 108
        self.before_first_meal = True

    def create_horizon_scenario(self, sample, states, patient_idx, length):
        """ Creates scenario for horizon simulation.
        
        Args:
            sample (int): current sample.
            states (ndarray): patient states.
        
        Returns:
            Scenario: scenario for horizon simulation.
        
        """
        hor_scenario = deepcopy(self.scenario)
        hor_scenario.settings.start_time = 0
        hor_scenario.settings.end_time = sample + self.prediction_horizon
        binmap = np.asarray(hor_scenario.inputs.meal_carb.start_time[0])<sample
        meals_ctrl = np.asarray(hor_scenario.inputs.meal_carb.magnitude[0])[binmap]
        meal_times_ctrl = np.asarray(hor_scenario.inputs.meal_carb.start_time[0])[binmap]
        meal_durations_ctrl = 15.0*np.ones_like(meal_times_ctrl)
        hor_scenario.inputs.meal_carb = Events([meals_ctrl], [meal_times_ctrl], [meal_durations_ctrl])
        hor_scenario.inputs.taud = generate_carb_absorption(hor_scenario,None)
        # hor_scenario.inputs.basal_insulin = Events([[0.0]], [[0.0]])
        # hor_scenario.inputs.bolus_insulin = Events([[0.0]], [[0.0]])
        hor_scenario.settings.sampling_time = self.ctrl_sampling_time

        return hor_scenario

    def run(self, sample, states, measured_glucose : float, patient_idx: int):
        """ Performs gradient descent algorithm to find optimal insulin input.
        
        Args:
            hor_scenario (Scenario): data needed for simulating the virtual patient.
            observer_simulation_data (SimulationData): contains estimated meal parameters.
            measured_glucose (float): last CGM measurement.
            observer_ran (bool): true if observer ran and patient and  meal parameters were updated.
        
        Returns:
            float: insulin value to be injected in the present moment.
        
        """
        hor_scenario = self.create_horizon_scenario(sample, states, patient_idx, self.prediction_horizon)
        self.solver = BaseSolver(hor_scenario, IVP.Model.from_scenario(hor_scenario))
        self.carb = np.copy(self.solver.model.inputs.carb.sampled_signal)
        self.taud = np.copy(self.solver.model.inputs.taud.sampled_signal)
        self.time = np.copy(self.solver.model.time.as_unix)

        inputs = self.solver.model.inputs
        # inputs.basal_insulin.sampled_signal = np.zeros((1,self.steps - sample))
        # inputs.bolus_insulin.sampled_signal = np.zeros((1,self.steps - sample))
        # inputs.carb.sampled_signal = self.carb[[0],sample : sample+self.steps]
        # inputs.taud.sampled_signal = self.taud[[0],sample : sample+self.steps]
        # self.solver.model.time.as_unix = self.time[sample : sample+self.steps]

        # self.solver.
        self.solver.model.preprocessing()
        self.basal_equilibrium = self.solver.model.get_basal_equilibrium(self.solver.model.parameters.as_array, self.glucose_init)
        # if self.before_first_meal:
        inputs.basal_insulin.sampled_signal[:, 0:sample-1] = self.basal_equilibrium
        self.insulin_horizon = self.basal_equilibrium #np.zeros(self.control_horizon)
        self.solver.model.initial_conditions.as_array = self.solver.model.output_equilibrium(self.solver.model.parameters.as_array, inputs.as_array)
        
        # self.plot_prediction(states, self.approx_x, patient_idx)
        
        # Set glucose target according to physical activity intensity
        # if physical_activity:
        #     self.glucose_target[self.iterator] = self.ideal_glucose + physical_activity * 15.0
        # else:
        self.glucose_target[self.iterator] = self.ideal_glucose
        
        # if self.iterator:
        #     self.set_bolus_insulin(hor_scenario, np.append(self.insulin_horizon, np.zeros(self.prediction_horizon - self.control_horizon)))
        
        # TODO: IVP output is normoglycemic without control (which is weird).
        self.approx_x = self.solver.do_simulation(True)
        self.approx_x_current = measured_glucose
        
        grad_counter = 0
        gradient_norm = 100 * self.epsilon
        start_time = timeit.default_timer()
        self.cost_array = np.zeros((self.max_grad_counter,))
        grad_array = np.zeros((self.max_grad_counter,))
        while grad_counter < self.max_grad_counter and gradient_norm > self.epsilon:
            # summed_arr = self.set_bolus_insulin(hor_scenario, np.append(self.insulin_horizon, np.zeros(self.prediction_horizon - self.control_horizon)))
            inputs.basal_insulin.sampled_signal[:,0] = self.insulin_horizon
            gradient, self.cost_array[grad_counter] = self.get_gradient(hor_scenario, self.solver, self.insulin_horizon, states, patient_idx)
            gradient_norm = np.linalg.norm(gradient)
            gradient = gradient * 10 ** 6
            grad_array[grad_counter] = gradient_norm
            if self.verbose:
                print(
                    "Step:" + str(grad_counter) + " Cost:" + str(self.cost_array[grad_counter]) + " Gradient norm: " + str(
                        gradient_norm))
        
            if not grad_counter or self.cost_array[grad_counter] < cost_optimal:
                cost_optimal = self.cost_array[grad_counter]
                insulin_optimal = np.copy(self.insulin_horizon)
        
            # Standard gradient descent
            self.insulin_horizon = self.insulin_horizon - self.grad_stepsize * gradient
            # Applying saturation
            self.insulin_horizon = np.clip(self.insulin_horizon, 0, self.insulin_limit)
            grad_counter += 1
        
        self.counter_array[self.iterator] = np.nonzero(self.cost_array < cost_optimal + 1.0)[0][0]
        
        elapsed_time = timeit.default_timer() - start_time
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear') # Clear console otherwise it gets cluttered
            print("Step:", (self.iterator + 1) * hor_scenario.settings.sampling_time, "/", self.steps, "   Elapsed time:", elapsed_time, "[s]")
        
        self.insulin[self.iterator] = insulin_optimal[0]
        
        self.set_bolus_insulin(hor_scenario, np.append(self.insulin[self.iterator], np.zeros(self.prediction_horizon - self.control_horizon)))
        # inputs.basal_insulin.sampled_signal[:,:] = insulin_optimal
        if self.verbose:
            print("CGM: ", measured_glucose, "  Insulin: ", insulin_optimal[0], "[uU/min]")
            print("First cost: ", self.cost_array[0], "Last cost: ", self.cost_array[-1], "Min cost: ",
              np.min(self.cost_array))
            print("--------------------------------------------------")
        
        # Simulate approximated patient in horizon
        self.state_init = self.approx_x_current
        self.approx_x = self.solver.do_simulation(True)#self.simulate_patient(hor_simulation_data)[1]
        self.approx_x_current = self.approx_x[patient_idx,0,0]
        
        self.cgms[self.iterator] = measured_glucose
        self.iterator += 1
        bolus_Uhr = UnitConversion.insulin.mUmin_to_Uhr(insulin_optimal[0])
        bolus_mUmin = UnitConversion.insulin.Uhr_to_mUmin(bolus_Uhr)

        # if sample > self.scenario.inputs.meal_carb.start_time[0][0]:
        #     self.before_first_meal = False
        print("\rBolus: ", bolus_Uhr, "Patient Basal: ", states[patient_idx, 0, 0], "IVP def Basal: ", inputs.basal_insulin.sampled_signal[:, 0:sample-1][0][-1], end="")
        # self.basal_rate += bolus_Uhr #insert bolus Uhr
        # inputs.bolus_insulin.sampled_signal[:, -1] = bolus_mUmin / 5
        # self.insulin_horizon = np.ones(self.control_horizon) * (bolus_mUmin / len(self.insulin_horizon) * 5)
        if any(meal_time < sample < meal_time + 5 for meal_time in self.scenario.inputs.meal_carb.start_time[0]):
            self.plot_prediction(states, self.approx_x, inputs.basal_insulin.sampled_signal[:, 0:sample-1], patient_idx)
    
        return bolus_mUmin, self.approx_x

    def quadratic_cost(self, glucose: np.ndarray, insulin: np.ndarray, patient_idx: int):
        """ Quadratic cost function: delivers asymmetric quadratic cost consisting of input and output costs.
    
                Args:
                    glucose (ndarray): blood glucose in horizon (-> output cost).
                    insulin (ndarray): insulin input in horizon (-> input cost).
    
                Returns:
                    float: quadratic cost for set point control.
        """
        cost = 0.0
        delta_hypo = 1.0
        delta_hyper = 20.0
        input_coeff_positive = 100
        input_coeff_negative = 10
        # Output cost (TODO: target should be a range not a single value. Being over 110 is not hyperglycemia)
        for i in range(len(glucose)):
            if glucose[i] <= self.glucose_target[self.iterator]:
                cost += ((self.glucose_target[self.iterator] - glucose[i]) / delta_hypo) ** 2
            elif glucose[i] > self.glucose_target[self.iterator]:
                cost += ((self.glucose_target[self.iterator] - glucose[i]) / delta_hyper) ** 2
        # Input cost
        # for ii in range(len(insulin)):
        #     if insulin[ii] <= self.insulin_init:
        #         cost += ((self.insulin_init - insulin[ii]) / 10 ** 6) ** 2 * input_coeff_negative
        #     elif insulin[ii] > self.insulin_init:
        #         cost += ((self.insulin_init - insulin[ii]) / 10 ** 6) ** 2 * input_coeff_positive
    
        return cost
    
    def get_gradient(self, scenario : scenario, solver : BaseSolver, insulin_in: np.ndarray, states, patient_idx: int):
        """ Calculates gradient of insulin vector.
    
            Args:
                insulin_in (ndarray): insulin input.
                scenario (scenario): data needed for simulating of the virtual patient.
    
            Returns:
                tuple[ndarray, float]: gradient of insulin vector, cost calculated with insulin_in.
        """
        gradient = np.zeros(self.control_horizon)
        shift = np.zeros(self.control_horizon)
    
        # Simulate glyc trajecory without bolus insulin injected, store cost to cost_in
        self.state_init = self.approx_x_current
        insulin_check = np.copy(insulin_in)
        self.set_bolus_insulin(scenario, np.append(insulin_in, np.zeros(self.prediction_horizon - self.control_horizon)))
        estimation = solver.do_simulation(True)[0][0]
        cost_in = self.quadratic_cost(UnitConversion.glucose.concentration_mmolL_to_mgdL(estimation), insulin_in, patient_idx) # TODO: pass measured_glucose, not state[:, 8] variable
    
        # Iterate over horizon and calc cost at each point
        for i in range(self.control_horizon):
            shift[i] = self.gradient_delta
            if i > 0:
                shift[i - 1] = 0.0

            # Simulate glyc trajectory with bolus (gradient_delta???) and compare costs: cost_shift - const_in and store gradient
            self.state_init = self.approx_x_current
            self.set_bolus_insulin(scenario, np.append(insulin_in + shift, np.zeros(self.prediction_horizon - self.control_horizon)))
            estimation = solver.do_simulation(True)[0][0]
    
            insulin_check[i] = insulin_check[i] + self.gradient_delta
            cost_shift = self.quadratic_cost(UnitConversion.glucose.concentration_mmolL_to_mgdL(estimation), insulin_check, patient_idx)
            gradient[i] = (cost_shift - cost_in) / self.gradient_delta
    
        # plt.show()
        return gradient, cost_in
    
    def set_bolus_insulin(self, scenario : scenario, insulin_in: np.ndarray):
        """ Sets bolus insulin values for virtual patient simulation.
    
                Args:
                    insulin_in (ndarray): insulin input.
    
                Returns:
                    int: number of insulin values outside of constraints.
    
        """
        scenario.inputs.bolus_insulin = insulin_in
        return np.sum(np.logical_or(insulin_in < 0.0, insulin_in > self.insulin_limit))
    
    def plot_prediction(self, states, prediction, ivp_basal, patient_idx):
        """ Plots prediction results.
    
        """
        import matplotlib
        matplotlib.use('MacOSX')
        plt.figure()
        plt.subplot(2, 1, 1)
        gluc = UnitConversion.glucose.concentration_mmolL_to_mgdL(states[patient_idx, 8, :])
        gluc = gluc[gluc > 0]
        plt.plot(gluc, label='Simulator Gluc.')
        horizon_time = np.linspace(len(gluc)-1, len(gluc)-1 + len(prediction[patient_idx, 0, :]) - 1, len(prediction[patient_idx, 0, :]))
        horizon_time = np.linspace(0, len(prediction[patient_idx, 0, :]) * 5, len(prediction[patient_idx, 0, :]))
        plt.plot(horizon_time, prediction[patient_idx, 0, :], label='IVP Gluc. (prediction)', linestyle='--')      
        if max(prediction[patient_idx, 0, :]) > 1000: # don't screw up the plot if the prediction is way off
            plt.ylim([min(gluc) - 10, max(gluc) + 10])
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        s1, I = states[patient_idx, 0, :], states[patient_idx, 2, :]
        plt.plot(s1[s1>0], label='Simulator S1')
        plt.plot(I[I>0], label='Simulator I')
        plt.plot(ivp_basal[0], label='IVP basal', linestyle='--')
        plt.grid()
        plt.legend()
        plt.show()
