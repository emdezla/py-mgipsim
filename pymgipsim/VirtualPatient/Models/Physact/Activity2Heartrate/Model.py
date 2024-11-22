from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim.Utilities.Timestamp import Timestamp
from .Parameters import Parameters
from .States import States
from .Inputs import Inputs
import numpy as np
from numba import njit


class Model(BaseModel, UnitConversion):
    name = "Physact.Activity2Heartrate"
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
        beta, VT_HRR, a, HRmax, HRrest, MaxSpeed, MaxGrade, BW, MAP, vGmax,\
            aL, aM, p0, r0, G0, M0, taoM, vMmax, GT, taoL, aG = parameters.T

        G, M, vL = states.T

        running_speed, running_incline, cycling_power, power, _ = inputs.T

        pbar = power / MAP
        pt = MAP * VT_HRR

        vLmax = 1.0 - vGmax
        vMM = vMmax * ((1.0 / (1.0 + np.exp(-(M - M0) / aM))))
        vLmax = vLmax - (vMM / 2.0)


        taor = (r0) * (1.0 - np.exp(-G0 / a)) / p0
        taop = (np.exp(1.0) - 1.0) / ((1.0 / taor) * (r0 + 60.0*(1.0 - np.exp(-1))) * (1.0 - np.exp(-GT / a)) - p0)

        DLp = vLmax * 2.0 * (1. / (1.0 + np.exp(-pbar / aL)) - 0.5)
        power_pt = np.exp(power / pt)
        fprod_p = power_pt - 1.0
        frem_p = r0 + 60.0*(1.0 - 1.0/power_pt)
        frem_G = 1.0 - np.exp(-G / a)
        frem_pG = frem_p * frem_G
        Gdotp = p0 + ((1.0 / taop) * fprod_p) - ((1.0 / taor) * frem_pG)
        Mdotp = 1.0 / taoM * (beta * pbar - M)
        vdotLp = (1.0 / taoL) * (DLp - vL)

        # binmap = np.logical_and(power < np.finfo(np.float64).eps, G < np.finfo(np.float64).eps)
        # Gdotp[binmap] = 0.0
        # Mdotp[binmap] = 0.0
        # vdotLp[binmap] = 0.0


        return np.column_stack((Gdotp, Mdotp, vdotLp))

    @staticmethod
    def rate_equations(states, time, parameters, inputs):
        G = states[:,0,:]
        M = states[:,1,:]
        vL = states[:,2,:]

        beta, VT_HRR, a, HRmax, HRrest, MaxSpeed, MaxGrade, BW, MAP, vGmax,\
            aL, aM, p0, r0, G0, M0, taoM, vMmax, GT, taoL, aG = parameters.T


        vMM = vMmax[:,None] * (1. / (1 + np.exp(-(M - M0[:,None]) / aM[:,None])))
        vG_max = vGmax[:,None] - (vMM / 2)
        vGG = vG_max* (1 - np.exp(-((G - G0[:,None])/ aG[:,None])))

        HRR = HRmax - HRrest
        vmin = HRrest / HRR
        v = vmin[:,None] + (vL + vGG + vMM)
        heart_rate = v * HRR[:,None]

        for rates,rest_rate in zip(heart_rate,HRrest):
            rates[rates<rest_rate] = rest_rate

        return heart_rate

    @staticmethod
    def output_equilibrium(parameters, inputs):
        pass

    def update_scenario(self, scenario):
        pass

    def preprocessing(self):
        self.initial_conditions.as_array = np.zeros((self.parameters.as_array.shape[0], 3))
        self.states.as_array = np.zeros(
            (self.inputs.as_array.shape[0], self.initial_conditions.as_array.shape[1], self.inputs.as_array.shape[2]))


        beta, VT_HRR, a, HRmax, HRrest, MaxSpeed, MaxGrade, BW, MAP, vGmax,\
            aL, aM, p0, r0, G0, M0, taoM, vMmax, GT, taoL, aG = self.parameters.as_array.T

        running_speed, running_incline, cycling_power, standard_power, _ = self.inputs.as_array.transpose([1,0,2])

        ASCMLim = 5 * 0.44704 * 60
        speed = running_speed * 0.44704 * 60
        grade = running_incline / 100.0
        METACSM = np.zeros_like(running_speed)
        binmap = np.logical_and(speed>np.finfo(np.float64).eps,speed < 3.5 * 0.44704 * 60)
        METACSM[binmap] = (0.1 * speed[binmap] + 1.8 * grade[binmap] * speed[binmap] + 3.5) / 3.5
        binmap = np.logical_and(speed >= 3.5 * 0.44704 * 60, speed < ASCMLim)
        METACSM[binmap] = ((1 - speed[binmap] / ASCMLim) * (0.1 * speed[binmap] + 1.8 * grade[binmap] * speed[binmap]) + speed[binmap] / ASCMLim * (
                    0.2 * speed[binmap] + 0.9 * grade[binmap] * speed[binmap])) / 3.5 + 1
        binmap = np.logical_and(speed >= 3.5 * 0.44704 * 60, speed >= ASCMLim)
        METACSM[binmap] = (0.2 * speed[binmap] + 0.9 * grade[binmap] * speed[binmap] + 3.5) / 3.5
        running_power = METACSM * 3.5 * 7

        binmap = running_power > MAP[:,None]
        running_power[binmap] = (MAP[:,None]*np.ones_like(binmap))[binmap]
        METACSM[binmap] = ((MAP[:,None]*np.ones_like(binmap))[binmap])/7/3.5

        binmap = cycling_power > MAP[:, None]
        METACSM[binmap] = ((10.8*cycling_power/(BW[:,None]*np.ones_like(binmap))+7.0)/3.5)[binmap]

        power = cycling_power + running_power

        # MAP = 0.9*MAP
        # beta = 1.1*beta
        limit = MAP*VT_HRR*1.1
        binmap = power>limit[:,None]
        power[binmap] = (limit[:,None]*np.ones_like(binmap))[binmap]

        self.inputs.standard_power.sampled_signal = power
        self.inputs.METACSM.sampled_signal = METACSM
        pass


