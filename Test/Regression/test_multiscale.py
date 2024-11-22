import unittest

from pymgipsim.Interface.parser import generate_parser_cli
from Test.Regression.model_test_base_class import CommonTests

class TestMultiscaleT2DModel(CommonTests.BaseTests):

    def setUp(self):

        default_args = generate_parser_cli().parse_args(['-ms', '-np', '-npb'])

        default_args_7_days = generate_parser_cli().parse_args(['-ms', '-d', '7', '-np', '-npb'])

        default_args_sglt2i_5mg = generate_parser_cli().parse_args(['-ms', '-sdm', '5', '-np', '-npb'])

        self.scenario_args_list = [('default_args', default_args), ('default_args_7_days', default_args_7_days), ('default_args_sglt2i_5mg', default_args_sglt2i_5mg)]


if __name__ == '__main__':
    unittest.main()