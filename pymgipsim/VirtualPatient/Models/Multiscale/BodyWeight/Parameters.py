import numpy as np
from ...Parameters import BaseParameters
from pymgipsim.Utilities.Scenario import scenario
from .CONSTANTS import *

class Parameters(BaseParameters):
    def __init__(self, parameters: np.ndarray = np.array([], dtype=float)):
        if parameters.size:
            (self.BW0,
            self.beta,
            self.EI0,
            self.EE0,
            self.rho,
            self.K) = parameters.T


    @property
    def as_array(self):
        parameters = [
                    self.BW0,
                    self.beta,
                    self.EI0,
                    self.EE0,
                    self.rho,
                    self.K
                    ]
        
        self._as_array = np.asarray(parameters).T
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        
        (self.BW0,
        self.beta,
        self.EI0,
        self.EE0,
        self.rho,
        self.K) = array

        self._as_array = np.asarray(array)

    @staticmethod
    def generate(scenario_instance: scenario):
        n_subjects = scenario_instance.patient.number_of_subjects

        BW0 = scenario_instance.patient.demographic_info.body_weight
        beta = np.full(n_subjects, NOMINAL_BETA)
        EI0 = np.full(n_subjects, NOMINAL_EI0)
        EE0 = np.full(n_subjects, NOMINAL_EE0)
        rho = np.full(n_subjects, NOMINAL_RHO)
        K = np.full(n_subjects, NOMINAL_K)

        parameter_array = np.column_stack((BW0, beta, EI0, EE0, rho, K))

        return parameter_array
