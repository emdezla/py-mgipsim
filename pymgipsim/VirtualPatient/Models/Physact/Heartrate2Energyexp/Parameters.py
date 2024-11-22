import numpy as np
import json
from ...Parameters import BaseParameters
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.paths import models_path
class Parameters(BaseParameters):

    def __init__(self, parameters: np.ndarray = np.array([], dtype=float)):
        self.as_array = np.empty((1,1),dtype=float)
        pass

    @property
    def as_array(self):
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        self._as_array = array

    @staticmethod
    def generate(scenario_instance: scenario):
        return

