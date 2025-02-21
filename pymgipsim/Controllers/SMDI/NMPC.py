import numpy as np
import timeit

from fontTools.misc.cython import returns
from matplotlib import pyplot as plt

from pymgipsim.InputGeneration.signal import Events
from pymgipsim.Utilities.Scenario import scenario
from copy import deepcopy
from pymgipsim.ModelSolver.BaseSolvers import BaseSolver
from pymgipsim.VirtualPatient.Models.T1DM import IVP
from pymgipsim.InputGeneration.carb_energy_settings import generate_carb_absorption

def euler(v, u_init, simulation_data):
    u_dot = - u_init * v
    return u_init + u_dot * simulation_data.scenario.Ts


class NMPC:
    """ Impulsive model predictive controller class with saturation.

    """

    def __init__(self, scenario: scenario):

        self.patient_idx = 0
        self.steps = 24
        self.ideal_glucose = 110
        self.glucose_target = np.zeros(self.steps)
        # Create containers
        self.insulin = np.zeros(self.steps)
        self.positive_insulin = np.zeros(self.steps)
        self.cgms = np.zeros(self.steps)

        self.multiple_taud = False
        self.verbose = True
        self.count_meals = 0
        self.iterator = 0
        self.total_cost = 0
        self.counter_array = np.zeros((self.steps,))
        self.control_horizon = 1
        self.prediction_horizon = 40
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
        self.scenario.patient.demographic_info.body_weight = [self.scenario.patient.demographic_info.body_weight[0]]
        self.scenario.patient.model.parameters = IVP.Parameters.generate(self.scenario)
        self.scenario.inputs.taud = generate_carb_absorption(self.scenario,None)
        self.solver = BaseSolver(self.scenario, IVP.Model.from_scenario(self.scenario))
        self.carb = np.copy(self.solver.model.inputs.carb.sampled_signal)
        self.taud = np.copy(self.solver.model.inputs.taud.sampled_signal)
        self.time = np.copy(self.solver.model.time.as_unix)
        # self.model = IVP.Model.from_scenario(self.ctrl_scenario)

        # # Initialize arrays
        # self.approx_x_init = self.get_insulin_equilibrium(self.ideal_glucose)
        # self.approx_x_current = self.approx_x_init
        # self.insulin_init = self.patient_params.CI * self.approx_x_init[3]
        # print(self.insulin_init)

    def run(self, sample, glucose):
        inputs = self.solver.model.inputs
        inputs.basal_insulin.sampled_signal, inputs.bolus_insulin.sampled_signal = np.zeros((1,self.steps)),np.zeros((1,self.steps))
        inputs.carb.sampled_signal = self.carb[[0],sample:sample + self.steps]
        inputs.taud.sampled_signal = self.taud[[0],sample:sample + self.steps]
        self.solver.model.time.as_unix = self.time[sample:sample + self.steps]
        self.solver.model.preprocessing()
        if sample==0:
            inputs.basal_insulin.sampled_signal[:,0] = self.solver.model.get_basal_equilibrium(self.solver.model.parameters.as_array, glucose)
            self.solver.model.initial_conditions.as_array = self.solver.model.output_equilibrium(self.solver.model.parameters.as_array, inputs.as_array)
        self.solver.do_simulation(False)
        return
        # """ Performs gradient descent algorithm to find optimal insulin input.
        #
        # Args:
        #     hor_scenario (Scenario): data needed for simulating the virtual patient.
        #     observer_simulation_data (SimulationData): contains estimated meal parameters.
        #     measured_glucose (float): last CGM measurement.
        #     observer_ran (bool): true if observer ran and patient and  meal parameters were updated.
        #
        # Returns:
        #     float: insulin value to be injected in the present moment.
        #
        # """
        #
        # # Set glucose target according to physical activity intensity
        # if physical_activity:
        #     self.glucose_target[self.iterator] = self.ideal_glucose + physical_activity * 15.0
        # else:
        #     self.glucose_target[self.iterator] = self.ideal_glucose
        #
        # if self.iterator:
        #     self.set_bolus_insulin(
        #         np.append(self.insulin_horizon, np.zeros(self.prediction_horizon - self.control_horizon)),
        #         hor_simulation_data, self.insulin_limit)
        #
        # self.approx_x_current[0] = measured_glucose
        # if self.iterator == 0:
        #     self.insulin_horizon = np.ones(self.control_horizon) * self.insulin_init
        #
        # grad_counter = 0
        # gradient_norm = 100 * self.epsilon
        # start_time = timeit.default_timer()
        # self.cost_array = np.zeros((self.max_grad_counter,))
        # grad_array = np.zeros((self.max_grad_counter,))
        # while grad_counter < self.max_grad_counter and gradient_norm > self.epsilon:
        #     summed_arr = self.set_bolus_insulin(
        #         np.append(self.insulin_horizon, np.zeros(self.prediction_horizon - self.control_horizon)),
        #         hor_simulation_data, self.insulin_limit)
        #     gradient, self.cost_array[grad_counter] = self.get_gradient(self.insulin_horizon, hor_simulation_data.copy())
        #     gradient_norm = np.linalg.norm(gradient)
        #     gradient = gradient * 10 ** 6
        #     grad_array[grad_counter] = gradient_norm
        #     print(
        #         "Step:" + str(grad_counter) + " Cost:" + str(self.cost_array[grad_counter]) + " Gradient norm: " + str(
        #             gradient_norm))
        #
        #     if not grad_counter or self.cost_array[grad_counter] < cost_optimal:
        #         cost_optimal = self.cost_array[grad_counter]
        #         insulin_optimal = np.copy(self.insulin_horizon)
        #
        #     # Standard gradient descent
        #     self.insulin_horizon = self.insulin_horizon - self.grad_stepsize * gradient
        #     # Applying saturation
        #     self.insulin_horizon = np.clip(self.insulin_horizon, 0, self.insulin_limit)
        #     grad_counter += 1
        #
        # self.counter_array[self.iterator] = np.nonzero(self.cost_array < cost_optimal + 1.0)[0][0]
        #
        # elapsed_time = timeit.default_timer() - start_time
        # if self.verbose:
        #     print("Step:", self.iterator + 1, "/", self.steps, "   Elapsed time:", elapsed_time, "[s]")
        #
        # self.insulin[self.iterator] = insulin_optimal[0]
        #
        # self.set_bolus_insulin(
        #     np.append(self.insulin[self.iterator], np.zeros(self.prediction_horizon - self.control_horizon)),
        #     hor_simulation_data, self.insulin_limit)
        #
        # if self.verbose:
        #     print("CGM: ", measured_glucose, "  Insulin: ", insulin_optimal[0], "[uU/min]")
        #     print("First cost: ", self.cost_array[0], "Last cost: ", self.cost_array[-1], "Min cost: ",
        #           np.min(self.cost_array))
        #     print("--------------------------------------------------")
        #
        # # Simulate approximated patient in horizon
        # self.state_init = self.approx_x_current
        # self.approx_x = self.simulate_patient(hor_simulation_data)[1]
        # self.approx_x_current = self.approx_x[1]
        #
        # self.cgms[self.iterator] = measured_glucose
        # self.iterator += 1
        #
        # return insulin_optimal[0], self.approx_x

    # def quadratic_cost(self, glucose: np.ndarray, insulin: np.ndarray):
    #     """ Quadratic cost function: delivers asymmetric quadratic cost consisting of input and output costs.
    #
    #             Args:
    #                 glucose (ndarray): blood glucose in horizon (-> output cost).
    #                 insulin (ndarray): insulin input in horizon (-> input cost).
    #
    #             Returns:
    #                 float: quadratic cost for set point control.
    #     """
    #     cost = 0.0
    #     delta_hypo = 1.0
    #     delta_hyper = 20.0
    #     input_coeff_positive = 100
    #     input_coeff_negative = 10
    #     # Output cost
    #     for ii in range(len(glucose)):
    #         if glucose[ii] <= self.glucose_target[self.iterator]:
    #             cost += ((self.glucose_target[self.iterator] - glucose[ii]) / delta_hypo) ** 2
    #         elif glucose[ii] > self.glucose_target[self.iterator]:
    #             cost += ((self.glucose_target[self.iterator] - glucose[ii]) / delta_hyper) ** 2
    #     # Input cost
    #     # for ii in range(len(insulin)):
    #     #     if insulin[ii] <= self.insulin_init:
    #     #         cost += ((self.insulin_init - insulin[ii]) / 10 ** 6) ** 2 * input_coeff_negative
    #     #     elif insulin[ii] > self.insulin_init:
    #     #         cost += ((self.insulin_init - insulin[ii]) / 10 ** 6) ** 2 * input_coeff_positive
    #
    #     return cost
    #
    # def get_gradient(self, insulin_in: np.ndarray, simulation_data: SimulationData):
    #     """ Calculates gradient of insulin vector.
    #
    #         Args:
    #             insulin_in (ndarray): insulin input.
    #             simulation_data (SimulationData): data needed for simulating of the virtual patient.
    #
    #         Returns:
    #             tuple[ndarray, float]: gradient of insulin vector, cost calculated with insulin_in.
    #     """
    #     gradient = np.zeros(self.control_horizon)
    #     shift = np.zeros(self.control_horizon)
    #
    #     self.state_init = self.approx_x_current
    #     insulin_check = np.copy(insulin_in)
    #     self.set_bolus_insulin(np.append(insulin_in, np.zeros(self.prediction_horizon - self.control_horizon)),
    #                          simulation_data, self.insulin_limit)
    #     self.simulate_patient(simulation_data)
    #     cost_in = self.quadratic_cost(self.state_historical[:, 0], insulin_in)
    #
    #     for ii in range(self.control_horizon):
    #         shift[ii] = self.gradient_delta
    #         if ii > 0:
    #             shift[ii - 1] = 0.0
    #
    #         self.state_init = self.approx_x_current
    #         self.set_bolus_insulin(
    #             np.append(insulin_in + shift, np.zeros(self.prediction_horizon - self.control_horizon)),
    #             simulation_data, self.insulin_limit)
    #         self.simulate_patient(simulation_data)
    #
    #         insulin_check[ii] = insulin_check[ii] + self.gradient_delta
    #         cost_shift = self.quadratic_cost(self.state_historical[:, 0], insulin_check)
    #         gradient[ii] = (cost_shift - cost_in) / self.gradient_delta
    #
    #     return gradient, cost_in
    #
    # def plot_result(self, virtual_patient: VirtualPatient, simulation_data: SimulationData, init_state):
    #     """ Plots results of the control simulation.
    #
    #         Args:
    #             virtual_patient (VirtualPatient): patient model.
    #             simulation_data (SimulationData): data needed for simulating the virtual patient.
    #             init_state (ndarray): initial state of the virtual patient at the start of simulation.
    #
    #     """
    #     plt.figure('BG-' + str(self.patient_idx))
    #     self.set_bolus_insulin(self.insulin, simulation_data, self.insulin_limit)
    #     virtual_patient.state_init = init_state
    #     virtual_patient.set_cgms_parameters(simulation_data.scenario, random_seed=1)
    #     virtual_patient.simulate_patient(simulation_data, with_cgms=True)
    #     plt.plot(simulation_data.time_range, virtual_patient.state_historical[:, 0],
    #              label="virtual patient without CGMS noise")
    #     plt.plot(simulation_data.time_range, self.cgms[:], linewidth=1.5,
    #              label="virtual patient with CGMS noise")
    #     min_glucose = np.min(simulation_data.glucose_level[:, 1])
    #     plt.vlines(simulation_data.meal.as_timestamped_array[:, 0],
    #                simulation_data.meal.as_timestamped_array[:, 1] - simulation_data.meal.as_timestamped_array[:,
    #                                                                  1] + min_glucose - 5,
    #                min_glucose - 5 + simulation_data.meal.as_timestamped_array[:, 1] * 2, linewidth=2,
    #                label="CH intake")
    #     plt.plot(simulation_data.time_range, self.glucose_target[:], linewidth=0.75, color='r', linestyle='-',
    #              label="glucose target")
    #     plt.xlabel("time [min]")
    #     plt.ylabel("blood glucose [mg/dl]")
    #     plt.legend()
    #     plt.savefig('SMC_tests\\BG-patient' + str(self.patient_idx) + '.png')
    #
    #     plt.figure('insulin-' + str(self.patient_idx))
    #     insulin = np.array([])
    #     for i in range(len(self.insulin)):
    #         insulin = np.append(insulin, np.ones(5) * self.insulin[i] * 5)
    #     self.plotted_insulin = insulin
    #     plt.plot(range(simulation_data.scenario.start_time.as_int, simulation_data.scenario.end_time.as_int + 5),
    #              insulin, linewidth=1.5, label="insulin")
    #     plt.axhline(y=self.insulin_init, color='r', linestyle='-', linewidth=0.75, label="basal insulin")
    #     plt.xlabel("time [min]")
    #     plt.ylabel("insulin [uU/min]")
    #     plt.legend()
    #     plt.savefig('SMC_tests\\insulin-patient' + str(self.patient_idx) + '.png')
    #     self.iterator -= 1
    #     self.total_cost = self.quadratic_cost(self.cgms, self.insulin)
    #
    # @staticmethod
    # def set_bolus_insulin(insulin_in: np.ndarray, simulation_data: SimulationData, insulin_limit):
    #     """ Sets bolus insulin values for virtual patient simulation.
    #
    #             Args:
    #                 insulin_in (ndarray): insulin input.
    #                 simulation_data (SimulationData): data needed for simulating a virtual patient.
    #                 insulin_limit (float): upper limit of insulin delivery per minute.
    #
    #             Returns:
    #                 int: number of insulin values outside of constraints.
    #
    #     """
    #     simulation_data.bolus.as_array = insulin_in
    #     return np.sum(np.logical_or(insulin_in < 0.0, insulin_in > insulin_limit))
