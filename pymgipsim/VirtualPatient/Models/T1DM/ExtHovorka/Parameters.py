import json

import numpy as np
from dataclasses import dataclass
from ...Parameters import BaseParameters
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.paths import models_path

from .CONSTANTS import *

class Parameters(BaseParameters):
    """ Stores the Hovorka model parameters.
            20 virtual patients available in the ./Patient folder.

            Hint:
                .as_array() function returns 2D numpy array where:
                1st dim: Subject in the virtual cohort
                2nd dim: Parameter


    """
    def __init__(self, parameters: np.ndarray = np.array([], dtype=float), number_of_subjects = 1):
        if parameters.size:
            (self.kb1, self.kb2, self.kb3, self.EGP0, self.ke, self.F01, self.AG, self.tmaxI, self.tmaxG,
             self.p3, self.p4, self.p5, self.beta, self.a, self.BW, self.HRrest, self.HRmax,
             self.VG, self.VI, self.VT_HRR,
             self.k12, self.ka1, self.ka2, self.ka3, self.aSI, self.b, self.c, self.dSI,
             self.tsub, self.p1, self.p2, self.p6, self.p7, self.p8, self.tmaxGFast) = parameters.T
        else:
            pass

    @property
    def as_array(self):
        parameters = [self.kb1, self.kb2, self.kb3, self.EGP0, self.ke, self.F01, self.AG, self.tmaxI, self.tmaxG,
             self.p3, self.p4, self.p5, self.beta, self.a, self.BW, self.HRrest, self.HRmax,
             self.VG, self.VI, self.VT_HRR,
             self.k12, self.ka1, self.ka2, self.ka3, self.aSI, self.b, self.c, self.dSI,
             self.tsub, self.p1, self.p2, self.p6, self.p7, self.p8, self.tmaxGFast]
        self._as_array = np.asarray(parameters).T
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        (self.kb1, self.kb2, self.kb3, self.EGP0, self.ke, self.F01, self.AG, self.tmaxI, self.tmaxG,
             self.p3, self.p4, self.p5, self.beta, self.a, self.BW, self.HRrest, self.HRmax,
             self.VG, self.VI, self.VT_HRR,
             self.k12, self.ka1, self.ka2, self.ka3, self.aSI, self.b, self.c, self.dSI,
             self.tsub, self.p1, self.p2, self.p6, self.p7, self.p8, self.tmaxGFast) = array.T
        self._as_array = np.asarray(array)


    @staticmethod
    def generate(scenario_instance: scenario):
        cohort_parameters = []
        path = models_path + "\\" + scenario_instance.patient.model.name.replace(".", "\\") + "\\Patients\\"
        for name in scenario_instance.patient.files:
            abs_path = path + name
            with open(abs_path) as f:
                params_dict = json.load(f)
            parameters = Parameters()
            parameters.fromJSON(params_dict["model_parameters"])
            cohort_parameters.append(parameters.as_array)
        return np.concatenate(cohort_parameters)
