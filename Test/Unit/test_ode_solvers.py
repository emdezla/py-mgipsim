import unittest, argparse
import numpy as np
from scipy.integrate import odeint
from pymgipsim.ODESolvers.ode_solvers import euler_single_step, rk4_single_step
def test_model_1(states, time, parameters, inputs):

	x1, x2 = states

	k1, k2 = parameters

	u = inputs

	dx1dt = u -k1 * x1

	dx2dt = k1 * x1 - k2 * x2

	diff = np.array([dx1dt.flatten(), dx2dt.flatten()]).flatten()

	return diff


class TestSolvers(unittest.TestCase):


	def setUp(self):

		self.valid_initial_conditions = np.array([10, 0])

		self.valid_parameters = np.array([5e-2, 5e-3])

		self.valid_time = np.arange(0, 300)

		self.valid_solution_no_inputs = odeint(func = test_model_1,
									y0 = self.valid_initial_conditions,
									t = self.valid_time,
									args = (self.valid_parameters,
											0,
											)
									)

		self.valid_inputs = np.zeros_like(self.valid_time)
		self.valid_inputs[np.argwhere(self.valid_time % 50 == 0).flatten()] = 10

		self.valid_solution = np.zeros_like(self.valid_solution_no_inputs)

		self.valid_solution[0] = self.valid_initial_conditions
		for i in range(1, self.valid_time.size):

			self.valid_solution[i, :] = odeint(func = test_model_1,
												y0 = self.valid_solution[i-1, :],
												t = [self.valid_time[i-1], self.valid_time[i]],
												args = (self.valid_parameters,
														self.valid_inputs[i],
														)
												)[-1, :]

	def test_valid_solutions_no_nan(self):
		self.assertFalse(np.isnan((self.valid_solution_no_inputs).any()))
		self.assertFalse(np.isnan((self.valid_solution).any()))

	def test_valid_solutions_no_negative(self):
		self.assertFalse(((self.valid_solution_no_inputs < 0).any()))
		self.assertFalse(((self.valid_solution < 0).any()))

	def test_euler(self):

		euler_sols_no_inputs = np.zeros_like(self.valid_solution_no_inputs)
		euler_sols_no_inputs[0] = self.valid_initial_conditions

		euler_sols = np.zeros_like(self.valid_solution_no_inputs)
		euler_sols[0] = self.valid_initial_conditions

		for i in range(1, self.valid_time.size):

			euler_sols_no_inputs[i, :] = euler_single_step(f = test_model_1,
																		h = 1,
																		initial = euler_sols_no_inputs[i-1, :],
																		time = self.valid_time[i-1],
																		parameters = self.valid_parameters,
																		inputs = 0
																		)

			euler_sols[i, :] = euler_single_step(f = test_model_1,
															h = 1,
															initial = euler_sols[i, :],
															time = self.valid_time[i-1],
															parameters = self.valid_parameters,
															inputs = self.valid_inputs[i-1]
															)


		self.assertTrue(np.allclose(euler_sols, self.valid_solution, rtol = 1, atol = 1))
		self.assertTrue(np.allclose(euler_sols_no_inputs, self.valid_solution_no_inputs, rtol = 1, atol = 1))


	def test_rk4(self):

		rk4_sols_no_inputs = np.zeros_like(self.valid_solution_no_inputs)
		rk4_sols_no_inputs[0] = self.valid_initial_conditions

		rk4_sols = np.zeros_like(self.valid_solution_no_inputs)
		rk4_sols[0] = self.valid_initial_conditions

		for i in range(1, self.valid_time.size):

			rk4_sols_no_inputs[i, :] = rk4_single_step(f = test_model_1,
																	h = 1,
																	initial = rk4_sols_no_inputs[i-1, :],
																	time = self.valid_time[i-1],
																	parameters = self.valid_parameters,
																	inputs = 0
																	)

			rk4_sols[i, :] = rk4_single_step(f = test_model_1,
															h = 1,
															initial = rk4_sols[i, :],
															time = self.valid_time[i-1],
															parameters = self.valid_parameters,
															inputs = self.valid_inputs[i-1]
															)


		self.assertTrue(np.allclose(rk4_sols, self.valid_solution, rtol = 1, atol = 1))
		self.assertTrue(np.allclose(rk4_sols_no_inputs, self.valid_solution_no_inputs, rtol = 1, atol = 1))



if __name__ == '__main__':
	unittest.main()


