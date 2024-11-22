import numpy as np
from dataclasses import dataclass, field
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models.Inputs import BaseInputs

@dataclass
class Inputs(BaseInputs):
    running_speed: Signal = field(default_factory=lambda: Signal())
    running_incline: Signal = field(default_factory=lambda: Signal())
    cycling_power: Signal = field(default_factory=lambda: Signal())
    standard_power: Signal = field(default_factory=lambda: Signal())
    METACSM: Signal = field(default_factory=lambda: Signal())

    @property
    def as_array(self):
        self._as_array = np.stack((self.running_speed.sampled_signal, self.running_incline.sampled_signal, self.cycling_power.sampled_signal,
                                   self.standard_power.sampled_signal, self.METACSM.sampled_signal),axis=1)
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        array_sw = np.swapaxes(array, 0, 1)
        self.running_speed.sampled_signal, self.running_incline.sampled_signal, self.cycling_power.sampled_signal,\
            self.standard_power.sampled_signal, self.METACSM.sampled_signal = array_sw
        self._as_array = array
