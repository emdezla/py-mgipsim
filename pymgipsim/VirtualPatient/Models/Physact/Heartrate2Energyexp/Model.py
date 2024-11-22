from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim.Utilities.Timestamp import Timestamp
from .Parameters import Parameters
from .States import States
from .Inputs import Inputs
import numpy as np
from .CONSTANTS import NOMINAL_tu1, NOMINAL_C
from numba import njit


class Model(BaseModel, UnitConversion):
    name = "Physact.Heartrate2Energyexp"
    output_state = 0

    def __init__(self, sampling_time):
        self.parameters = Parameters()
        self.inputs = Inputs()
        self.states = States()
        self.initial_conditions = States()
        self.time = Timestamp()
        self.sampling_time = sampling_time


    @staticmethod
    @njit("float64[:,:](float64[:,:],float64,float64[:,:],float64[:,:])", cache=True)
    def model(states, time, parameters, inputs):
        EE = states

        heart_rate, METACSM, deltaEE = inputs.T

        dotEE = (METACSM[:,None] + deltaEE[:,None] - EE) / NOMINAL_tu1

        return dotEE

    @staticmethod
    def rate_equations(states, time, parameters, inputs):
        pass

    @staticmethod
    def output_equilibrium(parameters, inputs):
        pass

    def update_scenario(self, scenario):
        pass

    def preprocessing(self):
        self.initial_conditions.as_array = np.zeros((self.inputs.as_array.shape[0], 1))
        self.states.as_array = np.zeros(
            (self.inputs.as_array.shape[0], self.initial_conditions.as_array.shape[1], self.inputs.as_array.shape[2]))

        heart_rate, METACSM, deltaEE = self.inputs.as_array.transpose([1,0,2])

        Carr = np.asarray(NOMINAL_C)
        Cext = Carr[:, 0:2][:, :, None, None]
        X = np.stack((METACSM / 10.0, heart_rate / 100.0))[None,:,:,:]
        dinv = 1.0/np.power(np.linalg.norm(Cext-X,axis=1),2)
        weight = dinv/np.sum(dinv,axis=0)

        deltaEE = np.squeeze(np.matmul(Carr[:,1][None,:],weight.transpose([1,0,2])),axis=1)

        binmap = METACSM < 1.0
        deltaEE[binmap] = 0.0

        self.inputs.deltaEE.sampled_signal = deltaEE
        pass


