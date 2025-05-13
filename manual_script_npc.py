import subprocess
from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder

from pymgipsim.Interface.parser import generate_parser_cli
from pymgipsim.InputGeneration.activity_settings import activity_args_to_scenario

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.generate_plots import generate_plots_main
from pymgipsim.generate_results import generate_results_main
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """ Parse Arguments  """
    args = generate_parser_cli().parse_args()

    """ Initialization """
    subprocess.run(['python', 'initialization.py'])

    """ Define Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

    """ Load Scenario """
    settings_file = simulation_folder.load_settings_file(args, results_folder_path)

    # Programatically define scenario
    args.controller_name = "MDI" # Select controller folder in pymgipsim/Controller/...
    args.model_name = "T1DM.ExtHovorka" # Select Hovorka model
    # args.patient_names = ["Patient_1"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.patient_names = ["Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5", "Patient_6", "Patient_7", "Patient_8", "Patient_9"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.running_speed = 0.0 # Turn off physical activity
    args.plot_patient = 0 # Plots patient glucose, intakes, heartrate
    args.random_seed = 0
    args.number_of_days = 7

    CI_range = range(12, 40)
    CF_range = range(30, 70)

    # Small range fot testing
    # CI_range = range(15, 18)
    # CF_range = range(40, 43)

    activity_args_to_scenario(settings_file, args)
    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

    # Hardcode random sequence of breakfast, lunch, and dinner carb values for 20 days
    hardcoded_breakfast_carb_values = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 40, 45, 50, 55, 60, 65, 70, 75]
    hardcoded_lunch_carb_values = [55, 65, 75, 85, 95, 45, 55, 65, 75, 85, 95, 50, 60, 70, 80, 90, 40, 50, 60, 70]
    hardcoded_dinner_carb_values = [70, 85, 100, 40, 55, 70, 85, 100, 40, 55, 70, 85, 100, 45, 60, 75, 90, 50, 65, 80]
    # Hardcode start_time arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_start_times = [400, 410, 420, 430, 440, 450, 460, 470, 480, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460]
    hardcoded_lunch_start_times = [730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 720, 730, 740, 750, 760, 770, 780, 790, 800]
    hardcoded_dinner_start_times = [1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160]
    # Hardcode duration arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_durations = [10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5]
    hardcoded_lunch_durations = [15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10]
    hardcoded_dinner_durations = [20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15]

    # Set the hardcoded values in the scenario for the specified number of days
    settings_file.inputs.meal_carb.magnitude = np.zeros((settings_file.patient.number_of_subjects, args.number_of_days * 3))
    settings_file.inputs.meal_carb.start_time = np.zeros((settings_file.patient.number_of_subjects, args.number_of_days * 3))
    settings_file.inputs.meal_carb.duration = np.zeros((settings_file.patient.number_of_subjects, args.number_of_days * 3))
    for j in range(settings_file.patient.number_of_subjects):
        for i in range(args.number_of_days):
            settings_file.inputs.meal_carb.magnitude[j][3 * i] = hardcoded_breakfast_carb_values[i]
            settings_file.inputs.meal_carb.magnitude[j][3 * i + 1] = hardcoded_lunch_carb_values[i]
            settings_file.inputs.meal_carb.magnitude[j][3 * i + 2 ] = hardcoded_dinner_carb_values[i]

            settings_file.inputs.meal_carb.start_time[j][3 * i] = hardcoded_breakfast_start_times[i]
            settings_file.inputs.meal_carb.start_time[j][3 * i + 1] = hardcoded_lunch_start_times[i]
            settings_file.inputs.meal_carb.start_time[j][3 * i + 2] = hardcoded_dinner_start_times[i]

            settings_file.inputs.meal_carb.duration[j][3 * i] = hardcoded_breakfast_durations[i]
            settings_file.inputs.meal_carb.duration[j][3 * i + 1] = hardcoded_lunch_durations[i]
            settings_file.inputs.meal_carb.duration[j][3 * i + 2] = hardcoded_dinner_durations[i]
        
    TIR_matrix = np.zeros((len(CI_range), len(CF_range), settings_file.patient.number_of_subjects))
    for i in CI_range:
        for j in CF_range:
            settings_file.patient.demographic_info.carb_insulin_ratio = [i] * settings_file.patient.number_of_subjects
            settings_file.patient.demographic_info.correction_bolus = [j] * settings_file.patient.number_of_subjects
            model,_ = generate_results_main(scenario_instance = settings_file, args = vars(args), results_folder_path = results_folder_path)
            
            # Calculate TIR (Time in Range) for all patients
            TIR = []
            for glucose_values in model.glucose:
                in_range = [70 <= glucose <= 180 for glucose in glucose_values]
                tir_percentage = sum(in_range) / len(glucose_values) * 100
                TIR.append(tir_percentage)

            # Store the average TIR for the current CI and CF in the matrix
            TIR_matrix[i - CI_range[0]][j - CF_range[0]] = TIR

            print(f"\rCI: {i}, CF: {j}, AVG TIR: {np.mean(TIR):.2f}%")
    

    # Convert CI_range and CF_range to numpy arrays
    CI_range_array = np.array(list(CI_range))
    CF_range_array = np.array(list(CF_range))

    # Convert TIR_matrix to a numpy array
    TIR_matrix_array = np.array(TIR_matrix)

    # Create a meshgrid for plotting
    CF, CI = np.meshgrid(CF_range_array, CI_range_array)

    # Create a single figure with separate 3D subplots for each patient
    num_patients = TIR_matrix_array.shape[2]
    fig = plt.figure()
    fig.canvas.manager.full_screen_toggle() # Fullscreen
    # Set the figure to full screen
    # fig.set_size_inches(18.5, 10.5, forward=True)
    for patient_idx in range(num_patients):
        rows = int(np.ceil(np.sqrt(num_patients)))
        cols = int(np.ceil(num_patients / rows))
        ax = fig.add_subplot(rows, cols, patient_idx + 1, projection='3d')
        surf = ax.plot_surface(CF, CI, TIR_matrix_array[:, :, patient_idx], cmap='viridis', alpha=0.7)

        # Add labels and title for each subplot
        ax.set_xlabel('Correction Factor (CF)')
        ax.set_ylabel('Carb Insulin Ratio (CI)')
        ax.set_zlabel('Time in Range (TIR)')
        ax.set_title(f'TIR Surface Plot for Patient {patient_idx + 1}')

        # Add a color bar for each subplot
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(TIR_matrix_array[:, :, patient_idx])
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()