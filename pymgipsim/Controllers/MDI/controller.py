from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.VirtualPatient import Models
import numpy as np
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models import T1DM
from copy import deepcopy

class Controller:
    name = "MDI"
    def __init__(self, scenario_instance: scenario):
        time = np.arange(scenario_instance.settings.start_time,scenario_instance.settings.end_time,scenario_instance.settings.sampling_time)
        self.model_name = scenario_instance.patient.model.name
        self.scenario = deepcopy(scenario_instance)
        self.add_meal_error = False

        # Random error in meal announcements
        if self.add_meal_error:
            rng = np.random.default_rng(42)
            ground_truth_meals = np.asarray(self.scenario.inputs.meal_carb.magnitude)
            # Draw 5 samples from a normal distribution with mean=0 and std=1
            # Very rough estimation from https://www.liebertpub.com/doi/10.1089/dia.2019.0502
            samples = rng.normal(loc=0.0, scale=1.0, size=ground_truth_meals.shape)
            self.scenario.inputs.meal_carb.magnitude = ground_truth_meals + ground_truth_meals/5.0*samples
            self.scenario.inputs.meal_carb.magnitude[self.scenario.inputs.meal_carb.magnitude<0.0] = 0.0

        match self.model_name:
            case T1DM.ExtHovorka.Model.name:
                self.to_mgdl = UnitConversion.glucose.concentration_mmolL_to_mgdL
                self.rate_to_uUmin = UnitConversion.insulin.mU_to_uU
                self.to_rate = lambda rate, bolus : UnitConversion.insulin.Uhr_to_mUmin(rate) + bolus
                self.insulin_idx = 3
                self.glucose_idx = 8
            case T1DM.IVP.Model.name:
                self.to_mgdl = lambda mgdl : mgdl
                self.rate_to_uUmin = lambda uU : uU
                self.to_rate = lambda rate, bolus : UnitConversion.insulin.Uhr_to_uUmin(rate) + UnitConversion.insulin.mU_to_uU(bolus)
                self.insulin_idx = 0
                self.glucose_idx = 0

        self.basal_rate = self.to_rate(np.asarray(self.scenario.patient.demographic_info.basal),0.0)

    def run(self, measurements, inputs, states, sample):

        insulin = []
        for patient_idx in range(len(measurements)):
            basal = self.scenario.patient.demographic_info.basal[patient_idx] * 0.9 # Basal rate is set 10% lower to be more realistic
            bolus_magnitude = 0.0

            measurements_mgdl = self.to_mgdl(measurements[patient_idx])

            meal_starts = np.array(self.scenario.inputs.meal_carb.start_time[patient_idx])
            binmap = np.logical_and(meal_starts <= sample * self.scenario.settings.sampling_time,
                                    sample * self.scenario.settings.sampling_time < meal_starts + self.scenario.settings.sampling_time)
            # Next meal announced
            if np.any(binmap):
                carb_insulin_ratio = self.scenario.patient.demographic_info.carb_insulin_ratio[patient_idx]
                correction_bolus = self.scenario.patient.demographic_info.correction_bolus[patient_idx]
                carb = np.asarray(self.scenario.inputs.meal_carb.magnitude[patient_idx])[binmap]
                bolus_magnitude = UnitConversion.insulin.U_to_mU(carb[0]/carb_insulin_ratio + max((measurements_mgdl - 120)/correction_bolus,0.0))

            insulin.append(self.to_rate(basal,bolus_magnitude/self.scenario.settings.sampling_time))
        inputs[:, self.insulin_idx, sample] = np.asarray(insulin)
        return