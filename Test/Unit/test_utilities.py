import unittest, os
from pymgipsim.Utilities.paths import simulator_path, scenarios_path, models_path, default_settings_path, results_path

class TestPaths(unittest.TestCase):

    def setUp(self):
        self.paths_list = [simulator_path, scenarios_path, models_path, default_settings_path, results_path]

    def test_defined_paths_exist(self):
        for p in self.paths_list:
            assert os.path.exists(p), f"path {p} doesn't exist"


if __name__ == '__main__':
    unittest.main()