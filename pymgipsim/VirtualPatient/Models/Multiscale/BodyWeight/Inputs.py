import numpy as np
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models.Inputs import BaseInputs
from dataclasses import dataclass, field

@dataclass
class Inputs(BaseInputs):
    energy_intake: Signal = field(default_factory=lambda: Signal())
    energy_expenditure: Signal = field(default_factory=lambda: Signal())
    urinary_glucose_excretion: Signal = field(default_factory=lambda: Signal())

    @property
    def as_array(self):
        self._as_array = np.stack((self.energy_intake.sampled_signal, self.energy_expenditure.sampled_signal, self.urinary_glucose_excretion.sampled_signal), axis=1)
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        self._as_array = array