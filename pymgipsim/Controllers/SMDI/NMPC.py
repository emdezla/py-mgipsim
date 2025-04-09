import numpy as np
import timeit
import os

from fontTools.misc.cython import returns
from matplotlib import pyplot as plt

from pymgipsim.InputGeneration.signal import Events, Signal
from pymgipsim.Utilities.Scenario import scenario
from copy import deepcopy
from pymgipsim.ModelSolver.BaseSolvers import BaseSolver
from pymgipsim.VirtualPatient.Models.T1DM import IVP
from pymgipsim.VirtualPatient.Models.T1DM.IVP import Parameters
from pymgipsim.InputGeneration.carb_energy_settings import generate_carb_absorption
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
import matplotlib
try:
    matplotlib.use('MacOSX')
except:
    pass

class NMPC:
    """ Impulsive model predictive controller class with saturation.

    """

    def __init__(self, scenario: scenario, patient_idx: int):

        self.patient_idx = 0
        self.ctrl_sampling_time = 5
        self.steps = int((scenario.settings.end_time - scenario.settings.start_time) / scenario.settings.sampling_time)
        self.ideal_glucose = 110
        self.glucose_target_range = [70, 180]
        # Create containers

        self.verbose = True
        self.use_built_in_plot = True
        self.iterator = 0
        self.control_horizon = 1
        self.prediction_horizon = 300
        self.gradient_delta = 100.0
        self.max_grad_counter = 500
        self.epsilon = 1e-6
        self.grad_gamma = 0.9
        self.grad_stepsize = 0.01
        self.insulin_limit = 25e6  # hard constraint, 25 U

        self.scenario = deepcopy(scenario)
        self.scenario.patient.model.name = IVP.Model.name
        self.scenario.patient.number_of_subjects = 1
        self.scenario.patient.demographic_info.body_weight = [self.scenario.patient.demographic_info.body_weight[patient_idx]]
        self.scenario.patient.model.parameters = IVP.Parameters.generate(self.scenario)
        self.scenario.inputs.taud = generate_carb_absorption(self.scenario,None)
        self.basal_rate = self.scenario.patient.demographic_info.basal[patient_idx]
        self.announced_meal_starts = np.array(self.scenario.inputs.meal_carb.start_time[patient_idx])
        self.announced_meal_amounts = np.array(self.scenario.inputs.meal_carb.magnitude[patient_idx])
        self.carb_insulin_ratio = self.scenario.patient.demographic_info.carb_insulin_ratio[patient_idx]
        # self.model = IVP.Model.from_scenario(self.ctrl_scenario)

        # Hardcoded init values for testing before identification algoritm
        self.glucose_init = 108
        self.before_first_meal = True
        self.last_plot_est = []
        self.estimations = []
        self.bolus_time_range = [120, 120]

        # Time of next meal for wich the insulin bolus has already been injected
        self.next_meal_bolus_injected = False

    def create_horizon_scenario(self, sample, ivp_params):
        """ Creates scenario for horizon simulation.
        
        Args:
            sample (int): current sample.
            states (ndarray): patient states.
        
        Returns:
            Scenario: scenario for horizon simulation.
        
        """
        hor_scenario = deepcopy(self.scenario)
        hor_scenario.settings.start_time = sample
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
        hor_scenario.patient.model.parameters = ivp_params
        # hor_scenario.patient.model.parameters = Parameters(np.asarray(ivp_params))

        return hor_scenario

    def run(self, sample, states, measured_glucose : float, patient_idx: int, ivp_params : Parameters, ivp_last_state : np.ndarray):
        """ Performs gradient descent algorithm to find optimal insulin input.
        
        Args:
            hor_scenario (Scenario): data needed for simulating the virtual patient.
            observer_simulation_data (SimulationData): contains estimated meal parameters.
            measured_glucose (float): last CGM measurement.
            observer_ran (bool): true if observer ran and patient and  meal parameters were updated.
        
        Returns:
            float: insulin value to be injected in the present moment.
        
        """

        # The end: plot full simulation
        if self.use_built_in_plot and sample >= self.scenario.settings.end_time-5:
            self.plot_prediction(states, None, None, None, patient_idx)

        # If we have already injected bolus for the next meal, skip
        # if sample < self.next_meal_bolus_injected or measured_glucose < self.glucose_target_range[0]:
        #     return 0, None

        # Check if meal is out of range or we have already injected insulin or if glucose is too low
        if not any(meal_time - self.bolus_time_range[0] <= sample < meal_time + self.bolus_time_range[1] for meal_time in self.scenario.inputs.meal_carb.start_time[0]):
            self.next_meal_bolus_injected = False
            return 0, None
        if self.next_meal_bolus_injected or measured_glucose < self.glucose_target_range[0]:
            return 0, None
        
        hor_scenario = self.create_horizon_scenario(sample, ivp_params)
        self.solver = BaseSolver(hor_scenario, IVP.Model.from_scenario(hor_scenario))
        self.carb = np.copy(self.solver.model.inputs.carb.sampled_signal)
        self.taud = np.copy(self.solver.model.inputs.taud.sampled_signal)
        self.time = np.copy(self.solver.model.time.as_unix)

        inputs = self.solver.model.inputs
        self.solver.model.initial_conditions.as_array = ivp_last_state

        self.glucose_init = measured_glucose

        self.solver.model.preprocessing()
        self.basal_equilibrium = self.solver.model.get_basal_equilibrium(self.solver.model.parameters.as_array, self.glucose_init)
        
        inputs.basal_insulin.sampled_signal[:, 0:sample-1] = self.basal_equilibrium

        bolus_insulin = 0
        self.solver.model.initial_conditions.as_array = self.solver.model.output_equilibrium(self.solver.model.parameters.as_array, inputs.as_array)
        
        grad_counter = 0
        gradient_norm = 100 * self.epsilon
        start_time = timeit.default_timer()
        cost_array = np.zeros((self.max_grad_counter,))
        gradi_array = np.zeros((self.max_grad_counter,))
        grad_array = np.zeros((self.max_grad_counter,))
        self.estimations = []

        while grad_counter < self.max_grad_counter and gradient_norm > self.epsilon:
            gradient, cost_array[grad_counter] = self.get_gradient(bolus_insulin, inputs)
            gradient_norm = np.linalg.norm(gradient)
            gradient = gradient * 10 ** 6
            gradi_array[grad_counter] = gradient
            grad_array[grad_counter] = gradient_norm
        
            if not grad_counter or cost_array[grad_counter] < cost_optimal:
                cost_optimal = cost_array[grad_counter]
                insulin_optimal = bolus_insulin
            else:
                pass
        
            # Gradient descent NOT used! Finding local minimum instead
            bolus_insulin = bolus_insulin + 10000 #- self.grad_stepsize * gradient
            # Applying saturation
            bolus_insulin = np.clip(bolus_insulin, 0, self.insulin_limit)
            grad_counter += 1
            # if self.verbose and self.cost_array[grad_counter] > 0 or gradient_norm > 0:
            #     print(
            #         "Step:" + str(grad_counter) + " Cost:" + str(self.cost_array[grad_counter]) + " Gradient norm: " + str(
            #             gradient_norm))

        # if gradient_norm > 0:
        #     plt.show()
        
        elapsed_time = timeit.default_timer() - start_time
        self.iterator += 1

        bolus_mUmin = insulin_optimal / 1000
        bolus_Uhr = UnitConversion.insulin.mUmin_to_Uhr(bolus_mUmin)

        # Reset IVP basal insulin and add injected insulin history
        inputs.basal_insulin.sampled_signal[:, 0:sample-1] = self.basal_equilibrium
        
        # Simulate approximated patient in horizon
        prediction = np.copy(self.solver.do_simulation(True))

        inputs.bolus_insulin.sampled_signal[:, 0] = insulin_optimal
        controlled_pred = np.copy(self.solver.do_simulation(True))

        # self.basal_rate += bolus_mUmin #insert bolus Uhr
        # inputs.bolus_insulin.sampled_signal[:, -1] = bolus_mUmin / 5

        if self.verbose and (np.min(cost_array) > 0 or bolus_Uhr > 0):
            print("Step:", (self.iterator + 1) * hor_scenario.settings.sampling_time, "/", self.steps, "   Elapsed time:", elapsed_time, "[s]")
            print("CGM: ", measured_glucose, "  Insulin opt.: ", insulin_optimal, "[uU/min]")
            print("First cost: ", cost_array[0], "Last cost: ", cost_array[-1], "Min cost: ", np.min(cost_array))
            print("Bolus: ", bolus_mUmin, "[mU/min] Patient Basal: ", states[patient_idx, 0, 0], "IVP def Basal: ", inputs.basal_insulin.sampled_signal[:, 0:sample-1][0][-1])
            print("--------------------------------------------------")

        # Used the bolus insulin for the next meal
        if bolus_mUmin > 0:
            self.next_meal_bolus_injected = True
        
            if self.use_built_in_plot:
                self.plot_prediction(states, prediction, controlled_pred, inputs.bolus_insulin.sampled_signal[0, :], patient_idx)
            
        return bolus_mUmin, prediction

    def quadratic_cost(self, glucose: np.ndarray):
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
        for i in range(len(glucose)):
            if glucose[i] <= self.glucose_target_range[0]:
                cost += ((self.ideal_glucose - glucose[i]) / delta_hypo) ** 2
            elif glucose[i] > self.glucose_target_range[1]:
                cost += ((self.ideal_glucose - glucose[i]) / delta_hyper) ** 2
    
        return cost
    
    def get_gradient(self, insulin_in: np.ndarray, inputs):
        """ Calculates gradient of insulin vector.
    
            Args:
                insulin_in (ndarray): insulin input.
                scenario (scenario): data needed for simulating of the virtual patient.
    
            Returns:
                tuple[ndarray, float]: gradient of insulin vector, cost calculated with insulin_in.
        """
        gradient = 0
        shift = 1
    
        # Simulate glyc trajecory with insulin_in injected, store cost to cost_in
        inputs.basal_insulin.sampled_signal[:, :] = self.basal_equilibrium
        self.set_bolus_insulin(insulin_in, inputs)
        gluc_estimation = self.solver.do_simulation(True)[0][0]
        cost_in = self.quadratic_cost(gluc_estimation)
        
        # Simulate glyc trajectory with insulin_in + shift and compare costs: cost_shift - const_in and store gradient
        self.set_bolus_insulin(insulin_in + shift, inputs)
        gluc_estimation = self.solver.do_simulation(True)[0][0]
        cost_shift = self.quadratic_cost(gluc_estimation)
        gradient = (cost_shift - cost_in) / self.gradient_delta
        if gradient != 0:
            # plt.plot(gluc_estimation)
            self.estimations.append(np.copy(gluc_estimation))
        return gradient, cost_in
    
    def set_bolus_insulin(self, insulin_in: np.ndarray, inputs):
        """ Sets bolus insulin values for virtual patient simulation.
    
                Args:
                    insulin_in (ndarray): insulin input.
        """
        inputs.basal_insulin.sampled_signal[:, 0] += insulin_in
        return
    
    def plot_prediction(self, states, prediction, controlled, ivp_basal, patient_idx):
        """ Plots prediction results.
    
        """
        plt.figure()
        plt.subplot(2, 1, 1)
        gluc = UnitConversion.glucose.concentration_mmolL_to_mgdL(states[patient_idx, 8, :])
        gluc = gluc[gluc > 0]
        plt.plot(gluc, label='Simulator Gluc.')
        plt.grid()
        plt.legend()

        # Plot glucose target range
        plt.axhline(self.glucose_target_range[0], color='red', linewidth=0.5)
        plt.axhline(self.glucose_target_range[1], color='green', linewidth=0.5)

        # Plot past estimations to compare with actual glucose
        if len(self.last_plot_est):
            for est in self.last_plot_est:
                plt.plot(est[0], est[1], linewidth=0.5, linestyle='--', color='blue', label='Past Estimation')

        # Plot prediction with and without control            
        if prediction is not None:
            horizon_time = np.linspace(len(gluc)-1, len(gluc)-1 + self.prediction_horizon, len(prediction[patient_idx, 0, :]))
            # horizon_time = np.linspace(0, len(prediction[patient_idx, 0, :]) * 5, len(prediction[patient_idx, 0, :]))
            plt.plot(horizon_time, prediction[patient_idx, 0, :], label='IVP Gluc. (prediction)', linestyle='--', color='red')
            for estimation in self.estimations:
                plt.plot(horizon_time, estimation, linewidth=0.3)
            plt.plot(horizon_time, controlled[patient_idx, 0, :], label='IVP Gluc. (controlled)', linestyle='--', color='green')

            # Store estimations
            self.last_plot_est.append([horizon_time, controlled[patient_idx, 0, :]])
            
            # Scale down plot if prediction is too high
            if max(prediction[patient_idx, 0, :]) > 2000:
                plt.ylim([min(gluc) - 10, max(gluc) + 10])

        # Plot meal times as arrows (dirac delta)
        meal_times = self.scenario.inputs.meal_carb.start_time[0]
        meal_magnitudes = self.scenario.inputs.meal_carb.magnitude[0]
        for meal_time, meal_magnitude in zip(meal_times, meal_magnitudes):
            if meal_time < len(gluc):  # Ensure meal time is within the plot range
                plt.arrow(meal_time, 0, 0, meal_magnitude, head_width=5, fc='black', ec='black')

        # Plot insulin
        plt.subplot(2, 1, 2)
        s1, I = states[patient_idx, 0, :], states[patient_idx, 2, :]
        plt.plot(s1[s1>0], label='Simulator S1')
        plt.plot(I[I>0], label='Simulator I')
        if prediction is not None:
            plt.plot(horizon_time, ivp_basal/1000, label='IVP basal', linestyle='--')
        # plt.plot(horizon_time, controlled[patient_idx, 2, :]/1000, label='IVP Ip', linestyle='--')
        # plt.plot(horizon_time, controlled[patient_idx, 3, :]/1000, label='IVP Isc', linestyle='--')
        plt.grid()
        plt.legend()
        plt.ylabel('Insulin [mU/min]')
        plt.show()
