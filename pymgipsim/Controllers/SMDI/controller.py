from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Controller:
    name = "SMDI"
    def __init__(self, scenario_instance):
        self.control_sampling = int(5/scenario_instance.settings.sampling_time)
        self.controllers = []
        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            self.controllers.append(NMPC(scenario_instance, patient_idx))

    def run(self, measurements, inputs, states, sample):
        if sample % self.control_sampling == 0:
            for patient_idx in range(inputs.shape[0]):
                bolus, gluc_pred = self.controllers[patient_idx].run(sample, states, UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[0]), patient_idx)
                inputs[patient_idx,3,sample:sample+self.control_sampling] = bolus
        return