from pymgipsim.Utilities.Scenario import load_scenario, save_scenario
from pymgipsim.Utilities.paths import default_settings_path, scenarios_path
from pymgipsim.Utilities.Scenario import inputs
from pymgipsim.InputGeneration.signal import Events
from pymgipsim.VirtualPatient.Models.T1DM.ExtHovorka.Parameters import Parameters
from pymgipsim.InputGeneration.heart_rate_settings import generate_heart_rate
from pymgipsim.InputGeneration.energy_expenditure_settings import generate_energy_expenditure
from pymgipsim.VirtualPatient.VirtualPatient import VirtualCohort


class RealTimeSimulator:

    def __init__(self, patient_name):
        self.settings_file = load_scenario(default_settings_path + "\\scenario_default.json")
        self.settings_file.patient.files = [patient_name]
        self.state = None

    def doSimulation(self,carbohydrate, insulin, step_time):
        self.settings_file.settings.end_time = 5
        self.settings_file.patient.number_of_subjects = 1
        self.settings_file.patient.model.parameters = Parameters.generate(self.settings_file)

        self.settings_file.inputs = inputs()
        self.settings_file.inputs.meal_carb = Events([[carbohydrate]],[[0]],[[5]])
        self.settings_file.inputs.snack_carb = Events([[0]],[[0]],[[5]])
        self.settings_file.inputs.running_speed = Events([[0]],[[0]],[[5]])
        self.settings_file.inputs.heart_rate, self.settings_file.inputs.METACSM = generate_heart_rate(self.settings_file, None)
        self.settings_file.inputs.energy_expenditure = generate_energy_expenditure(self.settings_file, None)
        self.settings_file.inputs.basal_insulin = Events([[insulin]],[[0]])
        self.settings_file.inputs.bolus_insulin = Events([[0]],[[0]],[[5]])

        cohort = VirtualCohort(self.settings_file)
        cohort.singlescale_model.preprocessing()

        if self.state is not None:
            cohort.singlescale_model.initial_conditions.as_array = self.state

        cohort.model_solver.do_simulation(True)

        model = cohort.singlescale_model
        # print(model.states.as_array.shape)
        self.state = model.states.as_array[:,:,-1]

        print(model.states.as_array[0,8,-1])


# realsim = RealTimeSimulator("Patient_1.json")
# for i in range(100):
#     realsim.doSimulation(0,1.1,0)
# print("")