from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from .States import States
from .Parameters import Parameters
from .Inputs import Inputs
from pymgipsim.Utilities.Timestamp import Timestamp
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.InputGeneration.signal import Signal
import numpy as np
from numba import njit

class Model(BaseModel, UnitConversion):
    name = "Multiscale.BodyWeight"
    glucose_state = None
    def __init__(self, sampling_time):
        self.inputs = Inputs()
        self.parameters = Parameters()
        self.states = States()
        self.initial_conditions = States()
        self.time = Timestamp()
        self.sampling_time = sampling_time


    def preprocessing(self):

        """
        This is going to function similar to 'initialize' the way I had it before. output_equilibrium does the initial condition
        calculation so this will just call that static method for this model.

        The values in self.initial_conditions can be assigned individually based on that the output_equilibrium (e.g. self.initial_conditions
        .Sg = 0)
        """

        self.initial_conditions.as_array = self.output_equilibrium(self.parameters.as_array, self.inputs.as_array)

        self.states.as_array = np.zeros((self.parameters.as_array.shape[0], self.initial_conditions.as_array.shape[1], self.time.as_unix.size))



    @staticmethod
    @njit(cache = True)
    def model(states, time, parameters, inputs) -> np.ndarray:

        BW = states.T

        EI, EE, UGE = inputs.T

        BW0, beta, EI0, EE0, rho, K = parameters.T
            
        dBWdt = ((EI - EI0) - (K + EE)*(BW - BW0)/(1-beta) - (EE - EE0)*BW0/(1-beta) - UGE)/rho
            
        return dBWdt.reshape(-1,1)

    @staticmethod
    def output_equilibrium(parameters: np.ndarray, inputs: np.ndarray) -> np.ndarray:

        """
        This calculates the steady state values for each state variable
        """
        BW0, _, _, _, _, _ = parameters.T

        return BW0.reshape(-1,1)

    @staticmethod
    def from_scenario(scenario_instance: scenario):
        time = np.arange(scenario_instance.settings.start_time,
                         scenario_instance.settings.end_time,
                         scenario_instance.settings.sampling_time
                         )

        model = Model(sampling_time=scenario_instance.settings.sampling_time)

        events = scenario_instance.inputs.energy_intake
        energy_intake = Signal(
                            time=time,
                            sampling_time=scenario_instance.settings.sampling_time,
                            duration=events.duration,
                            start_time=events.start_time,
                            magnitude=events.magnitude
                           )
        
        events = scenario_instance.inputs.energy_expenditure
        energy_expenditure = Signal(
                            time=time,
                            sampling_time=scenario_instance.settings.sampling_time,
                            duration=events.duration,
                            start_time=events.start_time,
                            magnitude=events.magnitude
                           )
        
        events = scenario_instance.inputs.urinary_glucose_excretion
        urinary_glucose_excretion = Signal(
                                        time=time,
                                        sampling_time=scenario_instance.settings.sampling_time,
                                        duration=events.duration,
                                        start_time=events.start_time,
                                        magnitude=events.magnitude
                                    )
        
        model.inputs = Inputs(
                            energy_intake=energy_intake,
                            energy_expenditure=energy_expenditure,
                            urinary_glucose_excretion=urinary_glucose_excretion
                            )
        
        model.parameters = Parameters(np.asarray(scenario_instance.patient.model.parameters))
        model.time = Timestamp()
        model.time.as_unix = time

        return model
    

    @staticmethod
    def from_scenario_multiscale(scenario_instance: scenario):

        """ Generate Parameters """
        model = Model(sampling_time=scenario_instance.settings.sampling_time)
        
        time = np.arange(
                        scenario_instance.settings.start_time,
                        UnitConversion.time.convert_minutes_to_days(scenario_instance.settings.end_time),
                        scenario_instance.settings.sampling_time
                        )


        """ Convert time to days """
        end_time_in_days = UnitConversion.time.convert_minutes_to_days(scenario_instance.settings.end_time)

        model = Model(sampling_time=scenario_instance.settings.sampling_time)

        sampling_time = np.full_like(scenario_instance.inputs.daily_energy_intake.start_time, scenario_instance.settings.sampling_time)

        energy_intake = Signal(
                            time=time,
                            sampling_time=scenario_instance.settings.sampling_time,
                            duration=sampling_time,
                            start_time=scenario_instance.inputs.daily_energy_intake.start_time,
                            magnitude=scenario_instance.inputs.daily_energy_intake.magnitude
                           )
        
        energy_expenditure = Signal(
                                    time=time,
                                    sampling_time=scenario_instance.settings.sampling_time,
                                    duration=sampling_time,
                                    start_time=scenario_instance.inputs.daily_energy_expenditure.start_time,
                                    magnitude=scenario_instance.inputs.daily_energy_expenditure.magnitude
                                    )
        
        urinary_glucose_excretion = Signal(
                                            time=time,
                                            sampling_time=scenario_instance.settings.sampling_time,
                                            duration=sampling_time,
                                            start_time=scenario_instance.inputs.daily_urinary_glucose_excretion.start_time,
                                            magnitude=scenario_instance.inputs.daily_urinary_glucose_excretion.magnitude
                                            )
        model.inputs = Inputs(
                            energy_intake=energy_intake,
                            energy_expenditure=energy_expenditure,
                            urinary_glucose_excretion=urinary_glucose_excretion
                            )
        
        model.parameters = Parameters(np.asarray(scenario_instance.patient.mscale.parameters))

        model.time = Timestamp()
        model.time.as_unix = time
        return model