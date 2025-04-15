import numpy as np

from pymgipsim.Utilities.Scenario import demographic_info
from ..SMDI.NMPC import NMPC
from ..SMDI.Estimation.estimator import Estimator
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models import T1DM

class Controller:
    name = "MPCPump"

    def __init__(self, scenario_instance):
        self.model_name = scenario_instance.patient.model.name
        self.scenario = scenario_instance
        self.control_sampling = int(5/scenario_instance.settings.sampling_time)
        self.controllers = [NMPC(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.estimators = [Estimator(scenario_instance, patient_idx) for patient_idx in range(scenario_instance.patient.number_of_subjects)]
        self.measurements = []
        self.insulins = []
        self.boluses = np.zeros((scenario_instance.patient.number_of_subjects, scenario_instance.settings.end_time // scenario_instance.settings.sampling_time))
        self.avg_carb_time = None
        [setattr(controller, 'control_horizon', 5) for controller in self.controllers]
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
                # if sample==UnitConversion.time.convert_hour_to_min(30):
                    # Estimate patient parameters based on 24h+ data
                    # estimator.run(sample, patient_idx, self.measurements, self.insulins)
                measurements_mgdl = self.to_mgdl(measurements[patient_idx])

                bolus = np.zeros((1,))
                binmap = np.logical_and(controller.announced_meal_starts <= sample,
                                        sample < controller.announced_meal_starts + 4)
                # Next meal announced
                if np.any(binmap):
                    bolus = UnitConversion.insulin.U_to_mU(controller.announced_meal_amounts[
                                                               binmap] / controller.carb_insulin_ratio / self.control_sampling)
                # Open-loop MDI therapy until patient parameters are not estimated
                if sample>=UnitConversion.time.convert_hour_to_min(30):
                    # (Re)create observer if not yet created or 2 hrs passed
                    if controller.ctrl_observer == None or sample % 120 == 0:
                        estimator.run(sample, patient_idx, self.measurements, self.insulins)
                        self.avg_carb_time = estimator.avg_carb_time
                        controller.create_contorller_observer(sample, estimator.scenario.patient.model.parameters, \
                                                              estimator.solver.model.states.as_array[0, :, -1], estimator.avg_carb_time)
                    observer_states = controller.simulate_observer(measurements_mgdl, sample) # Update observer states
                    # Call NMPC for bolus calculation
                    bolus, gluc_pred = controller.run(sample, states, measurements_mgdl,
                                                    patient_idx, controller.ctrl_observer.patient.model.parameters, observer_states[0, :, -1],
                                                        estimator.avg_carb_time)
                for i in range(bolus.shape[0]):
                    if bolus[i] > 0:
                        self.boluses[patient_idx, sample//self.control_sampling + i] = bolus[i]

                # Plot past predictions at last sample
                if sample >= self.scenario.settings.end_time - self.control_sampling and controller.use_built_in_plot:
                    controller.plot_prediction(states, None, None, None, patient_idx)

                inputs[patient_idx, self.insulin_idx, sample:sample + self.control_sampling] = self.to_rate(controller.basal_rate, self.boluses[patient_idx, sample//self.control_sampling])
                self.insulins.append(self.rate_to_uUmin(inputs[patient_idx, self.insulin_idx, sample]))
                self.measurements.append(measurements_mgdl)

        return