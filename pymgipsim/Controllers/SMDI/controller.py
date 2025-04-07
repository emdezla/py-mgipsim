from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from .Estimation.estimator import Estimator
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Controller:
    name = "SMDI"

    def __init__(self, scenario_instance):
        self.control_sampling = int(5/scenario_instance.settings.sampling_time)
        # self.nmpc = NMPC(scenario_instance)
        self.controllers = []
        self.estimators = []
        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            self.controllers.append(NMPC(scenario_instance, patient_idx))
            self.estimators.append(Estimator(scenario_instance, patient_idx))
        self.measurements = []
        self.insulins = []

    def run(self, measurements, inputs, states, sample):
        if sample % self.control_sampling == 0:

            for patient_idx in range(inputs.shape[0]):
                self.measurements.append(UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[patient_idx]))
                self.insulins.append(self.estimators[patient_idx].basal_insulin)

                # if len(self.measurements) == 96:
                #     self.estimators[patient_idx].run(sample, patient_idx, self.measurements, self.insulins)
                #     self.measurements.pop(0)
                #     self.insulins.pop(0)

                bolus, gluc_pred = self.controllers[patient_idx].run(sample, states, UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[patient_idx]), patient_idx)
                inputs[patient_idx,3,sample:sample+self.control_sampling] = UnitConversion.insulin.Uhr_to_mUmin(self.controllers[patient_idx].basal_rate)
        return