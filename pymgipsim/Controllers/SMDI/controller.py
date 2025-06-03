import numpy as np
import random

from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from .Estimation.estimator import Estimator
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models import T1DM

class Controller:
    name = "SMDI"

    def __init__(self, scenario_instance):
        self.model_name = scenario_instance.patient.model.name
        self.scenario = scenario_instance
        self.mdi_interval = 48
        self.add_meal_error = False
        self.control_sampling = int(5/scenario_instance.settings.sampling_time)
        # Random error in meal announcements
        if self.add_meal_error:
            rng = np.random.default_rng(42)
            ground_truth_meals = np.asarray(self.scenario.inputs.meal_carb.magnitude)
            # Draw 5 samples from a normal distribution with mean=0 and std=1
            # Very rough estimation from https://www.liebertpub.com/doi/10.1089/dia.2019.0502
            samples = rng.normal(loc=0.0, scale=1.0, size=ground_truth_meals.shape)
            self.scenario.inputs.meal_carb.magnitude = ground_truth_meals + ground_truth_meals/5.0*samples
            self.scenario.inputs.meal_carb.magnitude[self.scenario.inputs.meal_carb.magnitude<0.0] = 0.0

        self.controllers = [NMPC(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.estimators = [Estimator(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.measurements = [[] for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.insulins = [[] for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.boluses = np.zeros((scenario_instance.patient.number_of_subjects, scenario_instance.settings.end_time // scenario_instance.settings.sampling_time))
        [setattr(controller, 'control_horizon', 5) for controller in self.controllers]
        [setattr(controller, 'assume_basal', True) for controller in self.controllers]

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


    def run(self, measurements, inputs, states, sample):
        if sample % self.control_sampling == 0:

            for patient_idx, controller, estimator in zip(range(len(self.controllers),), self.controllers, self.estimators):
                measurements_mgdl = self.to_mgdl(measurements[patient_idx])

                bolus = np.zeros((1,))
                basal = self.scenario.patient.demographic_info.basal[patient_idx]
                binmap = np.logical_and(controller.announced_meal_starts <= sample,
                                        sample < controller.announced_meal_starts + 4)

                if sample == UnitConversion.time.convert_hour_to_min(self.mdi_interval):
                    self.mdi_interval = sample/60.0
                    # Run estimator once
                    estimator.run(sample, patient_idx, self.measurements[patient_idx], self.insulins[patient_idx])
                    controller.ivp_carb_time = np.copy(estimator.avg_carb_time)
                    controller.ivp_params = np.copy(estimator.scenario.patient.model.parameters)
                    controller.ivp_last_state = np.copy(estimator.solver.model.states.as_array[0, :, -1])
                    controller.ivp_last_state_open_loop = np.copy(estimator.solver.model.states.as_array[0, :, -1])
                    controller.last_measurement = measurements_mgdl
                # Simulate 5 min with Observer
                if sample >= UnitConversion.time.convert_hour_to_min(self.mdi_interval):
                    controller.update_observer(measurements_mgdl, sample)

                # Next meal announced
                if np.any(binmap):
                    # Open-loop MDI therapy until patient parameters are not estimated
                    if sample <= UnitConversion.time.convert_hour_to_min(self.mdi_interval):
                        basal = self.scenario.patient.demographic_info.basal[patient_idx] * 0.9 # Basal rate is set 10% lower to be more realistic
                        bolus = UnitConversion.insulin.U_to_mU(controller.announced_meal_amounts[
                                                               binmap] / controller.carb_insulin_ratio / self.control_sampling 
                                                               + max((measurements_mgdl - 120)/controller.correction_bolus,0.0))
                    # SMDI therapy after patient parameters are estimated
                    if sample>=UnitConversion.time.convert_hour_to_min(self.mdi_interval):
                        # Call NMPC for bolus calculation
                        bolus, gluc_pred = controller.run(sample, states, measurements_mgdl, patient_idx)
                        basal = controller.basal_rate
                    for i in range(bolus.shape[0]):
                        if bolus[i] > 0:
                            self.boluses[patient_idx, sample//self.control_sampling + i] = bolus[i]

                # Plot past predictions at last sample
                if sample >= self.scenario.settings.end_time - self.control_sampling and controller.use_built_in_plot:
                    controller.plot_prediction(states, None, None, None, patient_idx, self.mdi_interval)

                inputs[patient_idx, self.insulin_idx, sample:sample + self.control_sampling] = self.to_rate(basal, self.boluses[patient_idx, sample//self.control_sampling])
                self.insulins[patient_idx].append(self.rate_to_uUmin(inputs[patient_idx, self.insulin_idx, sample]))
                self.measurements[patient_idx].append(measurements_mgdl)

        return