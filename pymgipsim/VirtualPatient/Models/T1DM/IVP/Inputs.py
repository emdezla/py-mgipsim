import numpy as np
from dataclasses import dataclass, field
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models.Inputs import BaseInputs

@dataclass
class Inputs(BaseInputs):
    basal_insulin: Signal = field(default_factory=lambda: Signal())
    bolus_insulin: Signal = field(default_factory=lambda: Signal())
    taud: Signal = field(default_factory=lambda: Signal())
    carb: Signal = field(default_factory=lambda: Signal())
    Ra: Signal = field(default_factory=lambda: Signal())


    @property
    def as_array(self):
        self._as_array = np.stack((self.basal_insulin.sampled_signal, self.bolus_insulin.sampled_signal, self.taud.sampled_signal, self.carb.sampled_signal, self.Ra.sampled_signal),axis=1)
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        array_sw = np.swapaxes(array, 0, 1)
        self.basal_insulin.sampled_signal, self.bolus_insulin.sampled_signal, self.taud.sampled_signal, self.carb.sampled_signal, self.Ra.sampled_signal = array_sw
        self._as_array = array