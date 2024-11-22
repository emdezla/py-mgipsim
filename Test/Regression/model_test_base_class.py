from pymgipsim.main import run_simulator_cli
import numpy as np
import unittest

class CommonTests:

    class BaseTests(unittest.TestCase):

        def test_all_glucose_positive(self):

            for scenario_idx, (scenario_name, scenario_args) in enumerate(self.scenario_args_list):

                settings_file, model, figures = run_simulator_cli(scenario_args)

                output_state = model.singlescale_model.output_state

                glucose = model.singlescale_model.states.as_array[:, output_state]

                assert np.all(glucose >= 0), f"{scenario_name} has states < 0"


        def test_all_states_feasible(self):

            for scenario_idx, (scenario_name, scenario_args) in enumerate(self.scenario_args_list):

                settings_file, model, figures = run_simulator_cli(scenario_args)

                states = model.singlescale_model.states.as_array

                assert not np.isnan(states).any(), f"{scenario_name} has states < 0"