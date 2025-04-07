from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.Timestamp import Timestamp

from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.InputGeneration.waves import create_square_wave

from .States import States
from .Parameters import Parameters
from .Inputs import Inputs
from pymgipsim.VirtualPatient.Models.Model import BaseModel

import numpy as np

class Model(BaseModel, UnitConversion):
    name = "T1DM.IVP"
    glucose_state = 0
    output_state = 0

    def __init__(self, sampling_time):
        self.inputs = Inputs()
        self.parameters = Parameters()
        self.states = States()
        self.initial_conditions = States()
        self.time = Timestamp()
        self.sampling_time = sampling_time

    def preprocessing(self):

        self.initial_conditions.as_array = np.zeros((self.parameters.as_array.shape[0],4))
        self.inputs.Ra.sampled_signal = np.zeros_like(self.inputs.basal_insulin.sampled_signal)

        self.initial_conditions.as_array= self.output_equilibrium(self.parameters.as_array, self.inputs.as_array)

        inputs = self.inputs.as_array


        self.states.as_array = np.zeros((inputs.shape[0],4,inputs.shape[2]))

        # Inputs
        basal_insulin, bolus_insulin, taud, carb = self.inputs.basal_insulin.sampled_signal, self.inputs.bolus_insulin.sampled_signal, self.inputs.taud.sampled_signal, self.inputs.carb.sampled_signal

        # Parameters
        BW, EGP, GEZI, SI, CI, tau1, tau2, p2, Vg = self.parameters.as_array.T

        Ra = np.zeros((inputs.shape[0],inputs.shape[2]))

        distributed_start_times = create_square_wave(self.time.as_unix,self.inputs.carb.start_time,
                           self.inputs.carb.duration,self.inputs.carb.start_time,
                           self.inputs.carb.sampling_time,with_duration=False)
        distributed_taud = create_square_wave(self.time.as_unix,self.inputs.carb.start_time,
                           self.inputs.carb.duration,self.inputs.taud.magnitude,
                           self.inputs.carb.sampling_time,with_duration=False)
        distributed_carbs = carb*self.inputs.carb.sampling_time

        for patientidx in range(inputs.shape[0]):
            carbi = distributed_carbs[patientidx,:]
            taudi = distributed_taud[patientidx,:]
            meal_times = distributed_start_times[patientidx][carbi>0]#self.time.as_unix[carbi>0]
            taudi = taudi[taudi>0]
            carbi = carbi[carbi>0]

            rate_of_appearance_time_array_full = np.transpose(np.linspace(self.time.as_unix[0]-meal_times,self.time.as_unix[-1]-meal_times,self.time.as_unix.shape[0]))
            rate_of_appearance_time_array_full[rate_of_appearance_time_array_full<0.0] = 0.0

            check = np.transpose(np.expand_dims(taudi,0))
            d_temp = np.divide(-rate_of_appearance_time_array_full, check)
            ds_temp = np.transpose(np.divide(800.0 * carbi / Vg[patientidx],np.power(taudi, 2)))
            e_temp = np.exp(d_temp)
            m1_temp = np.multiply(np.expand_dims(ds_temp,1),rate_of_appearance_time_array_full)
            m2_temp = np.multiply(m1_temp,e_temp)
            Ra[patientidx,:] = np.sum(m2_temp,0)

        self.inputs.Ra.sampled_signal = Ra


    @staticmethod
    def model(states, time, parameters, inputs) -> np.ndarray:
        # States
        G, Ieff, Ip, Isc = states.T

        # Parameters
        BW, EGP, GEZI, SI, CI, tau1, tau2, p2, Vg = parameters.T

        # Inputs
        basal_insulin, bolus_insulin, taud, carb, Ra = inputs.T

        # Differential equations
        dIscdt = (-Isc + (bolus_insulin + basal_insulin) / CI) / tau1
        dIpdt = (-Ip + Isc) / tau2
        dIeffdt = p2 * (-Ieff + SI * Ip)
        dGdt = (- (GEZI + Ieff) * G + EGP + Ra)

        return np.column_stack((dGdt,dIeffdt,dIpdt,dIscdt))

    @staticmethod
    def get_basal_equilibrium(parameters, basal_blucose):
        """ Calculates steady-state of the insulin concentrations.
            The steady-state is calculated based on the glucose level and parameters.
            Zero CHO is assumed.
        """
        G0 = basal_blucose

        # Parameters
        BW, EGP, GEZI, SI, CI, tau1, tau2, p2, Vg = parameters.T

        Ieff0 = (EGP / G0) - GEZI
        if (Ieff0 < 0.0):
            Ieff0 = 0.0
        Ip0 = Ieff0 / SI
        Isc0 = Ip0
        basal_insulin = CI*Isc0

        return basal_insulin

    @staticmethod
    def output_equilibrium(parameters: np.ndarray, inputs: np.ndarray) -> np.ndarray:

        # Parameters
        BW, EGP, GEZI, SI, CI, tau1, tau2, p2, Vg = parameters.T

        # Inputs
        basal_insulin = inputs[:, 0, 0]

        Isc = np.multiply(np.divide(1.0, CI), basal_insulin)
        Ip = Isc
        Ieff= np.multiply(SI,Ip)
        G = np.divide(EGP, (GEZI + Ieff))

        return np.column_stack((G, Ieff, Ip, Isc))

    @staticmethod
    def rate_equations(states, time, parameters, inputs):
        pass

    @staticmethod
    def from_scenario(scenario_instance: scenario):

        time = np.arange(scenario_instance.settings.start_time,
                         scenario_instance.settings.end_time,
                         scenario_instance.settings.sampling_time
                         )
        
        model = Model(sampling_time=scenario_instance.settings.sampling_time)

        events = scenario_instance.inputs.basal_insulin
        converted = UnitConversion.insulin.Uhr_to_uUmin(np.asarray(events.magnitude))
        basal_insulin = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                                start_time=events.start_time, magnitude=converted)
        
        events = scenario_instance.inputs.bolus_insulin
        converted = UnitConversion.insulin.U_to_uU(np.asarray(events.magnitude))
        bolus_insulin = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                            start_time=events.start_time, magnitude=converted)
        
        events = scenario_instance.inputs.meal_carb
        meal_carb = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                        start_time=events.start_time, magnitude=events.magnitude)

        events = scenario_instance.inputs.taud
        meal_taud = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                        start_time=events.start_time, magnitude=events.magnitude)

        model.inputs = Inputs(basal_insulin=basal_insulin, bolus_insulin=bolus_insulin, taud=meal_taud, carb=meal_carb)

        model.parameters = Parameters(np.asarray(scenario_instance.patient.model.parameters))

        model.time = Timestamp()
        model.time.as_unix = time

        return model