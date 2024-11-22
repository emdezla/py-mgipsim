import unittest
import numpy as np
from pymgipsim.Probability.distributions import *

class TestUniformDistributions(unittest.TestCase):

    def setUp(self):
        self.lower_limit = 1
        self.upper_limit = 20
        self.valid_x = np.arange(self.lower_limit, self.upper_limit + 1)

    """ Uniform PDF """

    def test_uniform_valid(self):
        uniform_pdf(x = self.valid_x, lower = self.lower_limit, upper = self.upper_limit)

    def test_uniform_single_sample(self):
        uniform_pdf(x = self.valid_x, lower = self.lower_limit, upper=self.lower_limit)
        uniform_pdf(x = self.valid_x, lower = self.upper_limit, upper=self.upper_limit)

    def test_uniform_pdf_x(self):
        with self.assertRaises(AssertionError):
            uniform_pdf(x = [0, np.nan, 2], lower = self.lower_limit, upper = self.lower_limit - 5)

    def test_uniform_pdf_lower(self):
        with self.assertRaises(AssertionError):
            uniform_pdf(x = self.valid_x, lower = self.lower_limit, upper = self.lower_limit - 5)

    def test_uniform_pdf_upper(self):
        with self.assertRaises(AssertionError):
            uniform_pdf(x = self.valid_x, lower = self.upper_limit + 10, upper = self.upper_limit)


class TestTruncatedNormalDistribution(unittest.TestCase):

    def setUp(self):
        self.lower_limit = 1
        self.upper_limit = 20
        self.valid_x = np.arange(self.lower_limit, self.upper_limit + 1)

        self.mean = 10
        self.std = 2


    def test_truncated_normal_valid(self):
        truncated_normal_pdf(x = self.valid_x, mean = self.mean, std = self.std, lower = self.lower_limit, upper = self.upper_limit)

    def test_truncated_normal_pdf_x(self):
        with self.assertRaises(AssertionError):
            truncated_normal_pdf(x = [0, np.nan, 2], mean = self.mean, std = self.std, lower = self.lower_limit, upper = self.lower_limit - 5)

    def test_truncated_normal_pdf_lower(self):
        with self.assertRaises(AssertionError):
            truncated_normal_pdf(x = self.valid_x, mean = self.mean, std = self.std, lower = self.lower_limit, upper = self.lower_limit - 5)

    def test_truncated_normal_pdf_upper(self):
        with self.assertRaises(AssertionError):
            truncated_normal_pdf(x = self.valid_x, mean = self.mean, std = self.std, lower = self.upper_limit + 10, upper = self.upper_limit)

    def test_truncated_normal_pdf_mean(self):
        with self.assertRaises(AssertionError):
            truncated_normal_pdf(x = self.valid_x, mean = np.nan, std = self.std, lower = self.lower_limit, upper = self.upper_limit)

    def test_truncated_normal_pdf_std(self):
        with self.assertRaises(AssertionError):
            truncated_normal_pdf(x = self.valid_x, mean = self.mean, std = np.nan, lower = self.lower_limit, upper = self.upper_limit)


class TestNormalDistribution(unittest.TestCase):

    def setUp(self):
        self.lower_limit = 1
        self.upper_limit = 20
        self.valid_x = np.arange(self.lower_limit, self.upper_limit + 1)

        self.mean = 10
        self.std = 2

    def test_normal_valid(self):
        normal_pdf(x = self.valid_x, mean = self.mean, std = self.std)

    def test_normal_pdf_x(self):
        with self.assertRaises(AssertionError):
            normal_pdf(x = [0, np.nan, 2], mean = self.mean, std = self.std)

    def test_normal_pdf_mean(self):
        with self.assertRaises(AssertionError):
            normal_pdf(x = self.valid_x, mean = np.nan, std = self.std)

    def test_normal_pdf_std(self):
        with self.assertRaises(AssertionError):
            normal_pdf(x = self.valid_x, mean = self.mean, std = np.nan)

        with self.assertRaises(AssertionError):
            normal_pdf(x = self.valid_x, mean = self.mean, std = -5)

if __name__ == '__main__':
    unittest.main()