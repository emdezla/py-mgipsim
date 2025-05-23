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
from pymgipsim.VirtualPatient.Models import T1DM
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
        self.ideal_glucose = 100
        self.glucose_target_range = [80, 140]
        self.hypo_hyper_range = [70, 180]
        # Create containers

        self.verbose = True
        self.use_built_in_plot = True
        # self.control_horizon = 5 # Single injection works fine
        self.control_horizon = 30 # Change control horizon time to use single/multiple injection. (1 injection / controller sampling time)
        self.prediction_horizon = 360
        self.max_grad_counter = 500
        self.epsilon = 1e-3
        self.grad_stepsize = 10000
        self.grad_max_stepsize = 2e5 # 0.2 U/min
        self.insulin_limit = 25e6  # hard constraint, 25 U/min

        self.model_name = scenario.patient.model.name
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
        self.create_observer_scenario(scenario, patient_idx)
        # Hardcoded init values for testing before identification algoritm
        self.glucose_init = 108
        self.before_first_meal = True
        self.past_est_plots = []
        self.past_boluses = []
        self.estimations = []

        # Select to use target range or ideal glucose as target value
        self.use_target_range = False
        self.assume_basal = True
        self.check_settings()

        self.observer_preds = np.zeros(0,)
        self.observer_preds_openloop = np.zeros(0,)
        self.observer_insulin = np.zeros(0,)
        self.last_measurement = 0
        self.ivp_last_state = np.zeros(0,)
        self.ivp_last_state_open_loop = np.zeros(0, )
        self.ivp_params = np.zeros(0,)
        self.ivp_carb_time = 40

    def check_settings(self):
        """ Assertations for controller settings.
        
        """
        assert self.control_horizon > 0, "Control horizon must be greater than 0."
        assert self.control_horizon <= self.prediction_horizon, "Control horizon must be less than or equal to prediction horizon."
        assert self.control_horizon % self.ctrl_sampling_time == 0, "Control horizon must be a multiple of the control sampling time."
    
    def create_observer_scenario(self, fscenario : scenario, patient_idx : int):
        """ Creates scenario for observer simulation.
        
        Args:
            sample (int): current sample.
        
        """
        self.observer_scenario = deepcopy(fscenario)
        self.observer_scenario.patient.number_of_subjects = 1
        self.observer_scenario.patient.model.name = IVP.Model.name
        self.observer_scenario.patient.demographic_info.body_weight = [fscenario.patient.demographic_info.body_weight[patient_idx]]
        self.observer_scenario.patient.mscale.parameters = fscenario.patient.mscale.parameters[patient_idx]
        # Filter for patient_idx in basal and bolus insulin
        observer_inputs = self.observer_scenario.inputs
        scenario_inputs = fscenario.inputs
        try:
            observer_inputs.basal_insulin = Events([scenario_inputs.basal_insulin.magnitude[patient_idx]], [scenario_inputs.basal_insulin.start_time[patient_idx]], [scenario_inputs.basal_insulin.duration[patient_idx]])
        except IndexError:
            observer_inputs.basal_insulin = Events([scenario_inputs.basal_insulin.magnitude[patient_idx]], [scenario_inputs.basal_insulin.start_time[patient_idx]], [scenario_inputs.basal_insulin.duration])
            if observer_inputs.basal_insulin.duration.size == 0:
                observer_inputs.basal_insulin.duration = [np.array([0.0])]
        observer_inputs.bolus_insulin = Events([scenario_inputs.bolus_insulin.magnitude[patient_idx]], [scenario_inputs.bolus_insulin.start_time[patient_idx]], [scenario_inputs.bolus_insulin.duration[patient_idx]])

    def update_observer(self, measured_glucose, sample):
        self.observer_scenario.settings.start_time = sample - self.ctrl_sampling_time #360
        self.observer_scenario.settings.end_time = sample + 1
        binmap = (np.asarray(self.announced_meal_starts) < self.observer_scenario.settings.end_time) & \
             (np.asarray(self.announced_meal_starts) >= self.observer_scenario.settings.start_time - 60*10)
        meals_ctrl = np.asarray(self.announced_meal_amounts)[binmap]
        meal_times_ctrl = np.asarray(self.announced_meal_starts)[binmap]
        meal_durations_ctrl = 1.0*np.ones_like(meal_times_ctrl)
        observer_inputs = self.observer_scenario.inputs
        observer_inputs.meal_carb = Events([meals_ctrl], [meal_times_ctrl], [meal_durations_ctrl])
        observer_inputs.taud = generate_carb_absorption(self.observer_scenario,None, carb_time=self.ivp_carb_time)
        self.observer_scenario.settings.sampling_time = self.ctrl_sampling_time
        self.observer_scenario.patient.model.parameters = self.ivp_params


        observer_solver = BaseSolver(self.observer_scenario, IVP.Model.from_scenario(self.observer_scenario))
        observer_solver.model.preprocessing()
        observer_solver.model.initial_conditions.as_array = self.ivp_last_state_open_loop
        # observer_solver.model.initial_conditions.as_array[0] = self.last_measurement
        observer_solver.model.inputs.basal_insulin.sampled_signal[:, :] = UnitConversion.insulin.Uhr_to_uUmin(self.basal_rate)
        # Add past boluses to basal insulin sampled signal
        for bolus in self.past_boluses:
            if bolus[0] >= self.observer_scenario.settings.start_time and bolus[0] < self.observer_scenario.settings.end_time and bolus[1] > 0:
                observer_solver.model.inputs.basal_insulin.sampled_signal[:, (bolus[0] - self.observer_scenario.settings.start_time) // self.observer_scenario.settings.sampling_time] \
                    += UnitConversion.insulin.Uhr_to_uUmin(bolus[1])

        observer_states = np.copy(observer_solver.do_simulation(True))
        self.ivp_last_state_open_loop = np.copy(observer_states[0, :, -1])
        self.observer_preds_openloop = np.concatenate((self.observer_preds_openloop, observer_states[0, 0, 1:]))
        # self.observer_insulin = np.concatenate((self.observer_insulin, observer_states[0][2][1:]))
        # self.last_measurement = measured_glucose

        observer_solver = BaseSolver(self.observer_scenario, IVP.Model.from_scenario(self.observer_scenario))
        observer_solver.model.preprocessing()
        observer_solver.model.initial_conditions.as_array = self.ivp_last_state
        observer_solver.model.initial_conditions.as_array[0] = self.last_measurement
        observer_solver.model.inputs.basal_insulin.sampled_signal[:, :] = UnitConversion.insulin.Uhr_to_uUmin(self.basal_rate)
        # Add past boluses to basal insulin sampled signal
        for bolus in self.past_boluses:
            if bolus[0] >= self.observer_scenario.settings.start_time and bolus[0] < self.observer_scenario.settings.end_time and bolus[1] > 0:
                observer_solver.model.inputs.basal_insulin.sampled_signal[:, (bolus[0] - self.observer_scenario.settings.start_time) // self.observer_scenario.settings.sampling_time] \
                    += UnitConversion.insulin.Uhr_to_uUmin(bolus[1])

        observer_states = np.copy(observer_solver.do_simulation(True))
        self.ivp_last_state = np.copy(observer_states[0, :, -1])
        self.observer_preds = np.concatenate((self.observer_preds, observer_states[0, 0, 1:]))
        self.observer_insulin = np.concatenate((self.observer_insulin, observer_states[0][2][1:]))
        self.last_measurement = measured_glucose




    def create_horizon_scenario(self, sample, patient_idx):
        """ Creates scenario for horizon simulation.
        
        Args:
            sample (int): current sample.
            states (ndarray): patient states.
        
        """
        hor_scenario = deepcopy(self.scenario)
        hor_scenario.settings.start_time = sample
        hor_scenario.settings.end_time = sample + self.prediction_horizon
        binmap = (np.asarray(self.announced_meal_starts) < hor_scenario.settings.end_time) & \
             (np.asarray(self.announced_meal_starts) >= hor_scenario.settings.start_time - 60*10)
        # binmap = np.asarray(self.announced_meal_starts)<sample
        meals_ctrl = np.asarray(self.announced_meal_amounts)[binmap]
        meal_times_ctrl = np.asarray(self.announced_meal_starts)[binmap]
        meal_durations_ctrl = 15.0*np.ones_like(meal_times_ctrl)
        horizon_inputs = hor_scenario.inputs
        horizon_inputs.meal_carb = Events([meals_ctrl], [meal_times_ctrl], [meal_durations_ctrl])
        horizon_inputs.taud = generate_carb_absorption(hor_scenario,None, carb_time=self.ivp_carb_time) #MPCPump /w Hovorka: carb_time=55
        try:
            horizon_inputs.basal_insulin = Events([horizon_inputs.basal_insulin.magnitude[patient_idx]], [horizon_inputs.basal_insulin.start_time[patient_idx]], [horizon_inputs.basal_insulin.duration[patient_idx]])
        except IndexError:
            horizon_inputs.basal_insulin = Events([horizon_inputs.basal_insulin.magnitude[patient_idx]], [horizon_inputs.basal_insulin.start_time[patient_idx]], [horizon_inputs.basal_insulin.duration])
            if horizon_inputs.basal_insulin.duration.size == 0:
                horizon_inputs.basal_insulin.duration = [np.array([0.0])]
        horizon_inputs.bolus_insulin = Events([horizon_inputs.bolus_insulin.magnitude[patient_idx]], [horizon_inputs.bolus_insulin.start_time[patient_idx]], [horizon_inputs.bolus_insulin.duration[patient_idx]])

        hor_scenario.settings.sampling_time = self.ctrl_sampling_time
        hor_scenario.patient.model.parameters = self.ivp_params

        self.solver = BaseSolver(hor_scenario, IVP.Model.from_scenario(hor_scenario))
        self.carb = np.copy(self.solver.model.inputs.carb.sampled_signal)
        self.taud = np.copy(self.solver.model.inputs.taud.sampled_signal)
        self.time = np.copy(self.solver.model.time.as_unix)

        inputs = self.solver.model.inputs

        self.solver.model.preprocessing()
        print(self.solver.model.inputs.Ra.sampled_signal)

        if self.assume_basal:
            self.basal_equilibrium = self.solver.model.get_basal_equilibrium(self.solver.model.parameters.as_array, self.glucose_init)
            inputs.basal_insulin.sampled_signal[:, 0:sample-1] = self.basal_equilibrium

        self.solver.model.initial_conditions.as_array = np.copy(self.ivp_last_state)
        self.solver.model.initial_conditions.as_array[0] = self.glucose_init

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
        self.glucose_init = measured_glucose
        # Create horizon scenario
        self.create_horizon_scenario(sample, patient_idx)

        inputs = self.solver.model.inputs

        # Model Predictive Control
        start_time = timeit.default_timer()
        self.estimations = []
        cost_array = np.zeros((self.max_grad_counter,))

        insulin_optimal, grad_counter = self.gradient_descent(inputs, cost_array)
        
        elapsed_time = timeit.default_timer() - start_time

        bolus_mUmin = insulin_optimal / 1000
        bolus_Uhr = UnitConversion.insulin.mUmin_to_Uhr(bolus_mUmin)

        # Reset IVP basal insulin and add injected insulin history
        if self.assume_basal:
            inputs.basal_insulin.sampled_signal[:, 0:sample-1] = self.basal_equilibrium
        else:
            inputs.basal_insulin.sampled_signal[:, 0:sample-1] = 0
            self.basal_rate = np.zeros_like(self.basal_rate)
        
        # Simulate approximated patient in horizon
        prediction = np.copy(self.solver.do_simulation(True))

        self.set_bolus_insulins(insulin_optimal, inputs)
        controlled_pred = np.copy(self.solver.do_simulation(True))

        # self.basal_rate += bolus_mUmin #insert bolus Uhr
        # inputs.bolus_insulin.sampled_signal[:, -1] = bolus_mUmin / 5

        if self.verbose and np.any(bolus_Uhr) > 0:
            print("MPC prediction results-----------------------------------")
            print(f"Step: {sample}/{self.steps}   Elapsed time: {elapsed_time:.4f} [s] \033[92m CGM: {measured_glucose:.4f} [mg/dL] \033[0m")
            print(f"\033[91m Insulins: {bolus_Uhr} [U/hr] \033[0m")
            print(f"First cost: {cost_array[0]:.4f} Last cost: {cost_array[-1]:.4f} Min cost: {np.min(cost_array):.4f} Iterations: {grad_counter}")
            print(f"Patient Basal: {states[patient_idx, 0, 0]:.4f} IVP def Basal: {inputs.basal_insulin.sampled_signal[:, 0:sample-1][0][-1]:.4f}")  # Print array
            print("---------------------------------------------------------")

        if self.use_built_in_plot:
            # Save past predictions for plotting
            match self.model_name:
                case T1DM.ExtHovorka.Model.name:
                    gluc = UnitConversion.glucose.concentration_mmolL_to_mgdL(states[patient_idx, 8, :])
                case T1DM.IVP.Model.name:
                    gluc = states[patient_idx, 0, :]
            gluc = gluc[gluc > 0]
            horizon_time = np.linspace(len(gluc)-1, len(gluc)-1 + self.prediction_horizon, len(prediction[0, 0, :]))
            if np.any(bolus_Uhr) > 0:
                self.past_est_plots.append([horizon_time, controlled_pred[0, 0, :]])
            for i in range(len(bolus_Uhr)):
                self.past_boluses.append([sample + i * self.ctrl_sampling_time, bolus_Uhr[i]])
            # Plot current prediction
            # self.plot_prediction(states, prediction, controlled_pred, inputs.bolus_insulin.sampled_signal[0, :], patient_idx)
            
        return bolus_mUmin, prediction
    
    def gradient_descent(self, inputs, cost_array):
        num_of_bolus = int(self.control_horizon/self.ctrl_sampling_time)
        bolus_insulins = np.zeros((num_of_bolus,))
        grad_counter = 0
        gradient = self.epsilon + 1 
        while grad_counter < self.max_grad_counter and np.any(abs(gradient) - self.epsilon > 0) or grad_counter < 2:
            gradient, cost_array[grad_counter] = self.get_gradient(bolus_insulins, inputs)
            gradient = gradient * 10 ** -3
        
            if not grad_counter or cost_array[grad_counter] < cost_optimal:
                cost_optimal = cost_array[grad_counter]
                insulin_optimal = np.copy(bolus_insulins)
            
            # Gradient descent
            max_grad_idx = np.argmax(gradient)
            bolus_insulins[max_grad_idx] = bolus_insulins[max_grad_idx] + min(self.grad_max_stepsize, self.grad_stepsize / gradient[max_grad_idx])

            # Applying saturation
            bolus_insulins[max_grad_idx] = np.clip(bolus_insulins[max_grad_idx], 0, self.insulin_limit)
            grad_counter += 1
        return insulin_optimal, grad_counter

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
        delta_hyper = 10.0
        for i in range(len(glucose)):
            if self.use_target_range:
                if glucose[i] <= self.glucose_target_range[0]:
                    cost += ((self.ideal_glucose - glucose[i]) / delta_hypo) ** 2
                elif glucose[i] > self.glucose_target_range[1]:
                    cost += ((self.ideal_glucose - glucose[i]) / delta_hyper) ** 2
            else: # Use ideal glucose as target value
                if glucose[i] <= self.ideal_glucose:
                    cost += ((self.ideal_glucose - glucose[i]) / delta_hypo) ** 2
                elif glucose[i] > self.ideal_glucose:
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
        gradient = np.zeros((len(insulin_in),))
        shift = 100
    
        # Simulate glyc trajecory with insulin_in injected, store cost to cost_in
        if self.assume_basal:
            inputs.basal_insulin.sampled_signal[:, :] = self.basal_equilibrium
        else:
            inputs.basal_insulin.sampled_signal[:, :] = 0
        self.set_bolus_insulins(insulin_in, inputs)
        gluc_estimation = self.solver.do_simulation(True)[0][0]
        cost_in = self.quadratic_cost(gluc_estimation)

        # Simulate glyc trajectory with insulin_in + shift and compare costs: cost_shift - const_in and store gradient
        for i in range(len(insulin_in)):
            insulin_in[i] = insulin_in[i] + shift
            if i > 0:
                insulin_in[i-1] = insulin_in[i-1] - shift
            self.set_bolus_insulins(insulin_in, inputs)
            gluc_estimation = self.solver.do_simulation(True)[0][0]
            cost_shift = self.quadratic_cost(gluc_estimation)
            gradient[i] = (cost_in - cost_shift) / shift
        insulin_in[-1] = insulin_in[-1] - shift
        return gradient, cost_in
    
    def set_bolus_insulins(self, insulins_in: np.ndarray, inputs):
        """ Sets bolus insulin values for virtual patient simulation.
            Inject either a single bolus or a vector of boluses.
            If a vector is given, the length of the vector must match the length of the horizon.
    
                Args:
                    insulin_in (ndarray): insulin input.
                    inputs (Inputs): inputs for the virtual patient.
        """
        assert len(insulins_in) == 1 or len(insulins_in)*self.ctrl_sampling_time == self.control_horizon, \
        "Insulin input length is bigger than 1 but does not match control horizon length at current sampling rate."

        for i in range(0, len(insulins_in)):
            inputs.basal_insulin.sampled_signal[:,i] += insulins_in[i]
        return
    
    def plot_prediction(self, states, prediction, controlled, ivp_basal, patient_idx, obs_start):
        """ Plots prediction results.
    
        """
        plt.figure()
        plt.subplot(2, 1, 1)
        match self.model_name:
            case T1DM.ExtHovorka.Model.name:
                gluc = UnitConversion.glucose.concentration_mmolL_to_mgdL(states[patient_idx, 8, :])
            case T1DM.IVP.Model.name:
                gluc = states[patient_idx, 0, :]
        gluc = gluc[gluc > 0]
        plt.plot(gluc, label='Simulator Gluc.')
        obs_start = UnitConversion.time.convert_hour_to_min(obs_start)
        observer_time = np.linspace(obs_start, len(gluc)-1, len(self.observer_preds))
        plt.plot(observer_time, self.observer_preds, label='Observer Gluc.', linestyle='--', color='red')
        plt.plot(observer_time, self.observer_preds_openloop, label='Observer Gluc. OL', linestyle='--', color='orange')

        # Plot glucose hyper- and hypoglycemia levels
        plt.axhline(self.hypo_hyper_range[0], color='red', linewidth=0.5)
        plt.text(1, self.hypo_hyper_range[0], 'Hypoglycemia', color='red', fontsize=8)
        plt.axhline(self.hypo_hyper_range[1], color='green', linewidth=0.5)
        plt.text(1, self.hypo_hyper_range[1], 'Hyperglycemia', color='green', fontsize=8)

        # Plot glucose target range
        # plt.axhline(self.glucose_target_range[0], color='red', linestyle = '--', linewidth=0.5, alpha=0.5)
        # plt.text(1, self.glucose_target_range[0], 'Lower limit', color='red', fontsize=8)
        # plt.axhline(self.glucose_target_range[1], color='green', linestyle = '--', linewidth=0.5, alpha=0.5)
        # plt.text(1, self.glucose_target_range[1], 'Upper limit', color='green', fontsize=8)

        # Plot past estimations to compare with actual glucose
        if len(self.past_est_plots):
            for est in self.past_est_plots:
                plt.plot(est[0], est[1], linewidth=0.5, linestyle='--', color='grey')
            plt.plot(0, 0, linewidth=0.5, linestyle='--', color='grey', label='Past estimations')

        # Plot prediction with and without control            
        if prediction is not None:
            horizon_time = np.linspace(len(gluc)-1, len(gluc)-1 + self.prediction_horizon, len(prediction[patient_idx, 0, :]))
            # horizon_time = np.linspace(0, len(prediction[patient_idx, 0, :]) * 5, len(prediction[patient_idx, 0, :]))
            plt.plot(horizon_time, prediction[patient_idx, 0, :], label='IVP Gluc. (prediction)', linestyle='--', color='red')
            for estimation in self.estimations:
                plt.plot(horizon_time, estimation, linewidth=0.3)
            plt.plot(horizon_time, controlled[patient_idx, 0, :], label='IVP Gluc. (controlled)', linestyle='--', color='green')

            # Store estimations
            self.past_est_plots.append([horizon_time, controlled[patient_idx, 0, :]])
            
            # Scale down plot if prediction is too high
            if max(prediction[patient_idx, 0, :]) > 2000:
                plt.ylim([min(gluc) - 10, max(gluc) + 10])

        # Plot meal times as arrows (dirac delta)
        meal_times = self.scenario.inputs.meal_carb.start_time[0]
        meal_magnitudes = self.scenario.inputs.meal_carb.magnitude[0]
        for meal_time, meal_magnitude in zip(meal_times, meal_magnitudes):
            if meal_time < len(gluc):  # Ensure meal time is within the plot range
                plt.arrow(meal_time, 0, 0, meal_magnitude, head_width=5, fc='black', ec='black')
        plt.arrow(0, 0, 0, 0, fc='black', ec='black', label='Meals')

        # Plot past boluses
        if len(self.past_boluses):
            for bolus in self.past_boluses:
                if bolus[1] > 0:
                    plt.arrow(bolus[0], 0, 0, bolus[1], head_width=5, fc='red', ec='red')
            plt.arrow(0, 0, 0, 0, fc='red', ec='red', label='Boluses')
        
        plt.grid()
        plt.legend()

        # Plot insulin
        plt.subplot(2, 1, 2)
        s1 = states[patient_idx, 0, :]
        plt.plot(s1[s1>0], label='Simulator S1')
        plt.plot(observer_time, self.observer_insulin * 100, label='Observer Ip.', linestyle='--', color='red')
        if prediction is not None:
            plt.plot(horizon_time, ivp_basal/1000, label='IVP basal', linestyle='--')
        plt.grid()
        plt.legend()
        plt.ylabel('Insulin [mU/min]')
        #fig.canvas.manager.full_screen_toggle() # Fullscreen
        plt.show()
