from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from .Estimation.estimator import Estimator
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Controller:
    name = "SMDI"

    def __init__(self, scenario_instance):
        # self.nmpc = NMPC(scenario_instance)
        self.controllers = []
        self.estimators = []
        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            self.controllers.append(NMPC(scenario_instance, patient_idx))
            self.estimators.append(Estimator(scenario_instance, patient_idx))
        self.measurements = []
        self.insulins = []

    def run(self, measurements, inputs, states, sample):
        patient_idx = 0
        if sample % 5 == 0:
            self.measurements.append(UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[0]))
            self.insulins.append(self.estimators[patient_idx].basal_insulin)

            if len(self.measurements)==96:
                self.estimators[patient_idx].run(sample, patient_idx, self.measurements, self.insulins)
                self.measurements.pop(0)
                self.insulins.pop(0)

            # self.nmpc.run(sample, UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[0]))
            inputs[:, 3, :] = self.estimators[patient_idx].basal_insulin/1000.0
        return