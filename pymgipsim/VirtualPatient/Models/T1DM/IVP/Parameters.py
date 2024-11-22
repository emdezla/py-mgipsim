import numpy as np
from dataclasses import dataclass
from ...Parameters import BaseParameters
from pymgipsim.Utilities.Scenario import scenario
from .CONSTANTS import *
import json

class Parameters(BaseParameters):
    def __init__(self, parameters: np.ndarray = np.array([], dtype=float), number_of_subjects = 1):
        self.n_subjects = number_of_subjects
        if parameters.size:
            self.BW, self.EGP, self.GEZI, self.SI, self.CI, self.tau1, self.tau2, self.p2, self.Vg = parameters.T
        else:
            pass

    @property
    def as_array(self):
        parameters = [self.BW, self.EGP, self.GEZI, self.SI, self.CI, self.tau1, self.tau2, self.p2, self.Vg]
        self._as_array = np.asarray(parameters).T
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        self.BW, self.EGP, self.GEZI, self.SI, self.CI, self.tau1, self.tau2, self.p2, self.Vg = array.T
        self._as_array = np.asarray(array)


    @staticmethod
    def generate(scenario_instance: scenario):
        n_subjects = scenario_instance.patient.number_of_subjects
        parameter_array = np.asarray([np.asarray(scenario_instance.patient.demographic_info.body_weight),  # 0
                                      np.ones((n_subjects,)) * NOMINAL_EGP,  # 1
                                      np.ones((n_subjects,)) * NOMINAL_GEZI,  # 2
                                      np.ones((n_subjects,)) * NOMINAL_SI,  # 3
                                      np.ones((n_subjects,)) * NOMINAL_CI,  # 4
                                      np.ones((n_subjects,)) * NOMINAL_TAU1,  # 5
                                      np.ones((n_subjects,)) * NOMINAL_TAU2,  # 6
                                      np.ones((n_subjects,)) * NOMINAL_P2,  # 7
                                      np.ones((n_subjects,)) * NOMINAL_VG]).T
        return parameter_array
