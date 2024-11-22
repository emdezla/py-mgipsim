import numpy as np
from dataclasses import dataclass, field
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models.Inputs import BaseInputs

@dataclass
class Inputs(BaseInputs):
    heart_rate: Signal = field(default_factory=lambda: Signal())
    METACSM: Signal = field(default_factory=lambda: Signal())
    deltaEE: Signal = field(default_factory=lambda: Signal())

    @property
    def as_array(self):
        self._as_array = np.stack((self.heart_rate.sampled_signal, self.METACSM.sampled_signal, self.deltaEE.sampled_signal),axis=1)
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        array_sw = np.swapaxes(array, 0, 1)
        self.heart_rate.sampled_signal, self.METACSM.sampled_signal, self.deltaEE.sampled_signal = array_sw
        self._as_array = array
