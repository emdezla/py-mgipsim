import numpy as np
import json
from ...Parameters import BaseParameters
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.paths import models_path
class Parameters(BaseParameters):

    def __init__(self, parameters: np.ndarray = np.array([], dtype=float)):
        if parameters.size:
            (self.beta, self.VT_HRR, self.a, self.HRmax, self.HRrest, self.MaxSpeed,
             self.MaxGrade, self.BW, self.MAP, self.vGmax, self.aL, self.aM,
             self.p0, self.r0, self.G0, self.M0, self.taoM, self.vMmax, self.GT,
             self.taoL, self.aG) = parameters.T
        else:
            pass

    @property
    def as_array(self):
        parameters = [self.beta, self.VT_HRR, self.a, self.HRmax, self.HRrest, self.MaxSpeed,
             self.MaxGrade, self.BW, self.MAP, self.vGmax, self.aL, self.aM,
             self.p0, self.r0, self.G0, self.M0, self.taoM, self.vMmax, self.GT,
             self.taoL, self.aG]
        self._as_array = np.asarray(parameters).T
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        (self.beta, self.VT_HRR, self.a, self.HRmax, self.HRrest, self.MaxSpeed,
             self.MaxGrade, self.BW, self.MAP, self.vGmax, self.aL, self.aM,
             self.p0, self.r0, self.G0, self.M0, self.taoM, self.vMmax, self.GT,
             self.taoL, self.aG) = array.T
        self._as_array = np.asarray(array)


    @staticmethod
    def generate(scenario_instance: scenario):
        cohort_parameters = []
        path = models_path + "\\Physact\\Activity2Heartrate\\Patients\\"
        for name in scenario_instance.patient.files:
            abs_path = path + name
            with open(abs_path) as f:
                params_dict = json.load(f)
            model_params = params_dict["model_parameters"]
            parameters = Parameters()
            parameters.fromJSON(model_params)
            cohort_parameters.append(parameters.as_array)
        return np.concatenate(cohort_parameters)

