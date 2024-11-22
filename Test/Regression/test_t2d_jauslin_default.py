import unittest

from pymgipsim.Interface.parser import generate_parser_cli
from Test.Regression.model_test_base_class import CommonTests

class TestJauslinT2DModel(CommonTests.BaseTests):

    def setUp(self):

        default_args = generate_parser_cli().parse_args(['-np', '-npb'])

        default_args_7_days = generate_parser_cli().parse_args(['-mn', 'T2DM.Jauslin', '-d', '7', '-np', '-npb'])

        default_args_sglt2i_5mg = generate_parser_cli().parse_args(['-mn', 'T2DM.Jauslin', '-sdm', '5', '-np', '-npb'])

        self.scenario_args_list = [('default_args', default_args), ('default_args_7_days', default_args_7_days), ('default_args_sglt2i_5mg', default_args_sglt2i_5mg)]


if __name__ == '__main__':
    unittest.main()