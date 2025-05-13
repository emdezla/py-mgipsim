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
    args.running_speed = 0.0 # Turn off physical activity
    args.plot_patient = 0 # Plots patient glucose, intakes, heartrate
    args.breakfast_carb_range = [100, 101]
    args.am_snack_carb_range = [0, 0]
    args.lunch_carb_range = [100, 101]
    args.pm_snack_carb_range = [0, 0]
    args.dinner_carb_range = [100, 101]
    args.random_seed = 0

    CI_range = range(8, 58)
    CF_range = range(25, 75)

    activity_args_to_scenario(settings_file, args)
    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)
        
    TIR_matrix = np.zeros((len(CI_range), len(CF_range)))
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
            TIR_matrix[i - CI_range[0]][j - CF_range[0]] = np.mean(TIR)

            print(f"\rCI: {i}, CF: {j}, AVG TIR: {np.mean(TIR):.2f}%")
    

    # Convert CI_range and CF_range to numpy arrays
    CI_range_array = np.array(list(CI_range))
    CF_range_array = np.array(list(CF_range))

    # Convert TIR_matrix to a numpy array
    TIR_matrix_array = np.array(TIR_matrix)

    # Create a meshgrid for plotting
    CF, CI = np.meshgrid(CF_range_array, CI_range_array)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(CF, CI, TIR_matrix_array, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Correction Factor (CF)')
    ax.set_ylabel('Carb Insulin Ratio (CI)')
    ax.set_zlabel('Time in Range (TIR)')
    ax.set_title('TIR Surface Plot')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.show()