import numpy as np

from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from .Estimation.estimator import Estimator
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Controller:
    name = "SMDI"

    def __init__(self, scenario_instance):
        self.control_sampling = int(5/scenario_instance.settings.sampling_time)
        self.controllers = [NMPC(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.estimators = [Estimator(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.measurements = []
        self.insulins = []

    def run(self, measurements, inputs, states, sample):
        if sample % self.control_sampling == 0:

            for patient_idx, controller, estimator in zip(range(len(self.controllers),), self.controllers, self.estimators):
                if sample==UnitConversion.time.convert_hour_to_min(30):
                    # Estimate patient parameters based on 24h+ data
                    estimator.run(sample, patient_idx, self.measurements, self.insulins)

                bolus = 0
                binmap = np.logical_and(controller.announced_meal_starts <= sample,
                                        sample < controller.announced_meal_starts + 4)
                if np.any(binmap):
                    bolus = UnitConversion.insulin.U_to_mU(controller.announced_meal_amounts[
                                                               binmap] / controller.carb_insulin_ratio / self.control_sampling)
                # Open-loop MDI therapy until patient parameters are not estimated
                if sample>=UnitConversion.time.convert_hour_to_min(30):
                    # Call NMPC for bolus calculation
                    bolus, gluc_pred = controller.run(sample, states, UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[patient_idx]),
                                                       patient_idx, estimator.scenario.patient.model.parameters, estimator.solver.model.states.as_array[0, :, -1])
                inputs[patient_idx,3,sample:sample+self.control_sampling] = UnitConversion.insulin.Uhr_to_mUmin(controller.basal_rate) + bolus

                self.measurements.append(UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[patient_idx]))
                self.insulins.append(UnitConversion.insulin.mU_to_uU(inputs[patient_idx,3,sample]))
        return