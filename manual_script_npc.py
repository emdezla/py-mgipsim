from numba import jit, config
#config.DISABLE_JIT = True
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
from pymgipsim.Utilities.Scenario import scenario
from tqdm import tqdm
import json
from scipy.stats import norm
import csv
import os
import threading
from copy import deepcopy

def glycemic_risk_index(values):
    """
    Calculate the Glycemic Risk Index (GRI) for a list/Series of glucose values (mmol/L).
    GRI = LBGI + HBGI
    """
    f = (np.log(values) ** 1.084 - 5.381)
    lbgi = np.mean(np.where(f < 0, f, 0) ** 2) * 22.77
    hbgi = np.mean(np.where(f > 0, f, 0) ** 2) * 22.77
    return lbgi + hbgi

def parallel_run_simulation(i : int, settings_file : scenario, args, results_folder_path, GRI_matrix : np.ndarray, CI_range : range, CF_range : range):
    for j in tqdm(CF_range, disable = (not i == CI_range[0])):
        settings_file.patient.demographic_info.carb_insulin_ratio = [i] * settings_file.patient.number_of_subjects
        settings_file.patient.demographic_info.correction_bolus = [j] * settings_file.patient.number_of_subjects
        model,_ = generate_results_main(scenario_instance = settings_file, args = vars(args), results_folder_path = results_folder_path)
        
        # Calculate GRI (Glycemia Risk Index) for all patients
        GRI = []
        patient_idx = 1
        for glucose_values in model.glucose:
            gri = glycemic_risk_index(glucose_values)
            GRI.append(gri)
            patient_idx += 1

        # Store the average GRI for the current CI and CF in the matrix
        GRI_matrix[i - CI_range[0]][j - CF_range[0]] = GRI

def generate_or_read_meals(settings_file, args, csv_directory, meal_tir_stats : dict = None, generate_new_meals = False):

    # Hardcode random start_time arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_start_times = [400, 410, 420, 430, 440, 450, 460, 470, 480, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460]
    hardcoded_lunch_start_times = [730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 720, 730, 740, 750, 760, 770, 780, 790, 800]
    hardcoded_dinner_start_times = [1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160]
    # Hardcode random duration arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_durations = [10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5]
    hardcoded_lunch_durations = [15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10]
    hardcoded_dinner_durations = [20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15]

    # Iterate over each patient in the settings_file
    for patient_idx in range(settings_file.patient.number_of_subjects):

        # Save the generated meal_carb magnitudes into a CSV file
        csv_filename = os.path.join(csv_directory, f"patient_{patient_idx + 1}_meal_carb_magnitudes.csv")

        if generate_new_meals:
            # Get the mean and std for the current patient from the JSON file
            mean_meals_per_day = list(meal_tir_stats.values())[patient_idx]['mean_carb_intake']
            std_meals_per_day = list(meal_tir_stats.values())[patient_idx]['std_carb_intake']

            # Generate random meal values for each day such that the mean and std match the JSON values
            daily_meals = norm.rvs(loc=mean_meals_per_day, scale=std_meals_per_day, size=args.number_of_days)

            # Ensure no negative meal values
            daily_meals = np.abs(daily_meals)
            file = open(csv_filename, mode='w', newline='')
            writer = csv.writer(file)
            writer.writerow(["Day", "Meal", "Carb Magnitude"])
        else:
            # Load the meal carb magnitudes from the CSV file
            file = open(os.path.join(csv_filename), mode='r')
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                day = int(row[0])
                if day > args.number_of_days:
                    print(f"Stop reading meals at day {day} as it exceeds the number of simulated days.")
                    break
                meal = row[1]
                carb_magnitude = float(row[2])

                if meal == "Breakfast":
                    settings_file.inputs.meal_carb.magnitude[patient_idx][3 * (day - 1)] = carb_magnitude
                elif meal == "Lunch":
                    settings_file.inputs.meal_carb.magnitude[patient_idx][3 * (day - 1) + 1] = carb_magnitude
                elif meal == "Dinner":
                    settings_file.inputs.meal_carb.magnitude[patient_idx][3 * (day - 1) + 2] = carb_magnitude

        # Distribute the daily meal values across breakfast, lunch, and dinner
        for day_idx in range(args.number_of_days):
            if generate_new_meals:
                breakfast_ratio, lunch_ratio, dinner_ratio = np.random.dirichlet([1, 1, 1])
                breakfast_carb = daily_meals[day_idx] * breakfast_ratio
                lunch_carb = daily_meals[day_idx] * lunch_ratio
                dinner_carb = daily_meals[day_idx] * dinner_ratio

                settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx] = breakfast_carb
                settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx + 1] = lunch_carb
                settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx + 2] = dinner_carb
                
                writer.writerow([day_idx + 1, "Breakfast", settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx]])
                writer.writerow([day_idx + 1, "Lunch", settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx + 1]])
                writer.writerow([day_idx + 1, "Dinner", settings_file.inputs.meal_carb.magnitude[patient_idx][3 * day_idx + 2]])
                print(f"Patient {patient_idx + 1} daily carbs \nmean: generated: {np.mean(daily_meals):.2f} g in study: {mean_meals_per_day:.2f} g, \nstd: generated: {np.std(daily_meals):.2f} g in study: {std_meals_per_day:.2f} g")

            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx] = day_idx * 1440 + hardcoded_breakfast_start_times[day_idx]
            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx + 1] = day_idx * 1440 + hardcoded_lunch_start_times[day_idx]
            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx + 2] = day_idx * 1440 + hardcoded_dinner_start_times[day_idx]

            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx] = hardcoded_breakfast_durations[day_idx]
            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx + 1] = hardcoded_lunch_durations[day_idx]
            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx + 2] = hardcoded_dinner_durations[day_idx]
        # Close the CSV file
        file.close()

def generate_CI_CF_pairs(args, settings_file, results_folder_path):
    """
    Generate CI CF pairs for each patient based on meal statistics.
    """

    """ Load Scenario """
    settings_file = simulation_folder.load_settings_file(args, results_folder_path)

    generate_new_meals = False # Set to True to generate new meals, False to use previous from csv files
    resources_directory = "mpc_test"

    # Define paths for reading and writing resources
    meal_stats_path = os.path.join(resources_directory, 'meal_tir_stats.json')
    selected_pairs_path = os.path.join(results_folder_path, 'selected_CI_CF_pairs.json')
    surface_plot_path = os.path.join(results_folder_path, 'GRI_surface_plot.png')
    plots_directory = os.path.join(results_folder_path, "GRI_surface_plots")
    # gluc_plots_directory = os.path.join(resources_directory, "Glucose_plots")
    # Create a subdirectory for csv files
    csv_directory = os.path.join(resources_directory, "meal_carb_magnitudes")

    if meal_stats_path is not None and generate_new_meals:
        # Load meal statistics from mpc_meal_stats.json
        with open(meal_stats_path, 'r') as f:
            meal_tir_stats: dict = json.load(f)
    else:
        meal_tir_stats = None

    # Ensure the resources directories exist
    os.makedirs(resources_directory, exist_ok=True)
    os.makedirs(plots_directory, exist_ok=True)
    # os.makedirs(gluc_plots_directory, exist_ok=True)
    os.makedirs(csv_directory, exist_ok=True)

    CI_range = range(30, 70)
    CF_range = range(35, 65)

    # Small range fot testing
    # CI_range = range(18, 20)
    # CF_range = range(40, 43)

    activity_args_to_scenario(settings_file, args)
    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

    generate_or_read_meals(settings_file, args, csv_directory, meal_tir_stats, generate_new_meals)
        
    GRI_matrix = np.zeros((len(CI_range), len(CF_range), settings_file.patient.number_of_subjects))
    threads = []
    for i in CI_range:
        t = threading.Thread(
            target=parallel_run_simulation,
            args=(i, deepcopy(settings_file), args, results_folder_path, GRI_matrix, CI_range, CF_range)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Select CI, CF value pair for each patient
    selected_CI_CF_pairs = []
    for patient_idx in range(settings_file.patient.number_of_subjects):
        mean_gri = list(meal_tir_stats.values())[patient_idx]['mean_gri']

        # Find the CI, CF pair where GRI matches the mean_gri
        closest_match = None
        closest_diff = float('inf')
        for ci_idx, ci in enumerate(CI_range):
            for cf_idx, cf in enumerate(CF_range):
                gri_value = GRI_matrix[ci_idx, cf_idx, patient_idx]
                diff = abs(gri_value - mean_gri)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_match = (ci, cf)

        selected_CI_CF_pairs.append(closest_match)

    # Print the selected CI, CF pairs for each patient
    for patient_idx, (ci, cf) in enumerate(selected_CI_CF_pairs):
        print(f"Patient {patient_idx + 1}: Selected CI = {ci}, CF = {cf}, GRI from simulation = {GRI_matrix[ci - CI_range[0], cf - CF_range[0], patient_idx]:.2f}, Mean GRI from study = {list(meal_tir_stats.values())[patient_idx]['mean_gri']:.2f}")
    # Save the selected CI, CF pairs into a JSON file
    selected_CI_CF_dict = {
        list(meal_tir_stats.keys())[patient_idx]: {"Selected_CI": ci, "Selected_CF": cf}
        for patient_idx, (ci, cf) in enumerate(selected_CI_CF_pairs)
    }

    with open(selected_pairs_path, 'w') as json_file:
        json.dump(selected_CI_CF_dict, json_file, indent=4)

    # Run simulation with selected CI, CF settings for each patient and plot results
    print("\nRunning simulation with selected CI, CF settings for each patient...")

    # Set up a new scenario/settings for each patient with their selected CI, CF
    for patient_idx, (ci, cf) in enumerate(selected_CI_CF_pairs):
        # Set only the current patient's CI and CF, keep others unchanged
        settings_file.patient.demographic_info.carb_insulin_ratio[patient_idx] = ci
        settings_file.patient.demographic_info.correction_bolus[patient_idx] = cf

        # Run simulation for this patient only
        model, _ = generate_results_main(
            scenario_instance=settings_file,
            args=vars(args),
            results_folder_path=results_folder_path
        )
    figures = generate_plots_main(results_folder_path, args)

    # Convert CI_range and CF_range to numpy arrays
    CI_range_array = np.array(list(CI_range))
    CF_range_array = np.array(list(CF_range))

    # Convert GRI_matrix to a numpy array
    GRI_matrix_array = np.array(GRI_matrix)

    # Create a meshgrid for plotting
    CF, CI = np.meshgrid(CF_range_array, CI_range_array)

    # Create a single figure with separate 3D subplots for each patient
    num_patients = GRI_matrix_array.shape[2]
    fig = plt.figure()
    # fig.canvas.manager.full_screen_toggle() # Fullscreen
    # Set the figure to full screen
    # fig.set_size_inches(18.5, 10.5, forward=True)
    for patient_idx in range(num_patients):
        rows = int(np.ceil(np.sqrt(num_patients)))
        cols = int(np.ceil(num_patients / rows))
        ax = fig.add_subplot(rows, cols, patient_idx + 1, projection='3d')
        surf = ax.plot_surface(CF, CI, GRI_matrix_array[:, :, patient_idx], cmap='viridis', alpha=0.7)

        # Add labels and title for each subplot
        # ax.set_xlabel('Correction Factor (CF)')
        # ax.set_ylabel('Carb Insulin Ratio (CI)')
        # ax.set_zlabel('Glycemic Risk Index (GRI)')
        # ax.set_title(f'GRI Surface Plot for Patient {patient_idx + 1}')

        # Add a color bar for each subplot
        # mappable = plt.cm.ScalarMappable(cmap='viridis')
        # mappable.set_array(GRI_matrix_array[:, :, patient_idx])
        # fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(surface_plot_path, dpi=300)
    # plt.show()

    # Generate and save separate 3D plots for each patient
    for patient_idx in range(num_patients):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(CF, CI, GRI_matrix_array[:, :, patient_idx], cmap='viridis', alpha=0.7)

        # Add labels and title for each plot
        ax.set_xlabel('Correction Factor (CF)')
        ax.set_ylabel('Carb Insulin Ratio (CI)')
        ax.set_zlabel('Glycemic Risk Index (GRI)')
        ax.set_title(f'GRI Surface Plot for Patient {patient_idx + 1}')

        # Add a color bar for the plot
        # mappable = plt.cm.ScalarMappable(cmap='viridis')
        # mappable.set_array(GRI_matrix_array[:, :, patient_idx])
        # fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)

        # Save the plot to the subdirectory
        plot_filename = os.path.join(plots_directory, f'GRI_surface_plot_patient_{patient_idx + 1}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)

def call_simulation_with_CI_CF_pairs(args, old_results, settings_file = None):
    """
    Call the simulation with the selected CI and CF pairs for each patient.
    """
    # Load the selected CI, CF pairs from the JSON file
    selected_pairs_path = os.path.join(old_results, 'selected_CI_CF_pairs.json')
    with open(selected_pairs_path, 'r') as json_file:
        selected_CI_CF_dict = json.load(json_file)

    if settings_file is None:
        # Load settings from the JSON file
        settings_file = simulation_folder.load_settings_file(args, old_results)

    """ Define New Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

    # Iterate over each patient and run the simulation with their selected CI and CF
    activity_args_to_scenario(settings_file, args)
    settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)
    settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)
    settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)
    csv_directory = os.path.join("mpc_test", "meal_carb_magnitudes")
    generate_or_read_meals(settings_file, args, csv_directory, generate_new_meals = False)

    i :int = 0
    for patient_name, pairs in selected_CI_CF_dict.items():
        if patient_name in settings_file.patient.files:
            ci = pairs['Selected_CI']
            cf = pairs['Selected_CF']

            # Set the CI and CF for the current patient
            settings_file.patient.demographic_info.carb_insulin_ratio[i] = ci
            settings_file.patient.demographic_info.correction_bolus[i] = cf
            i += 1
    # Run the simulation
    model, _ = generate_results_main(scenario_instance=settings_file, args=vars(args), results_folder_path=results_folder_path)
    for patient_idx in range(settings_file.patient.number_of_subjects):
        args.plot_patient = patient_idx
        figures = generate_plots_main(results_folder_path, args)


if __name__ == '__main__':
    """ Parse Arguments  """
    args = generate_parser_cli().parse_args()

    """ Initialization """
    subprocess.run(['python', 'initialization.py'])

    # Programatically define scenario
    args.controller_name = "MDI" # Select controller folder in pymgipsim/Controller/...
    args.model_name = "T1DM.ExtHovorka" # Select Hovorka model
    args.patient_names = ["Patient_9"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    # args.patient_names = ["Patient_1", "Patient_2"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.running_speed = 0.0 # Turn off physical activity
    args.plot_all = True
    args.am_snack_carb_range = [0, 0] # Set to 0 to disable snack
    args.pm_snack_carb_range = [0, 0] # Set to 0 to disable snack 
    args.random_seed = 0
    args.number_of_days = 7
    args.no_progress_bar = True
    args.no_print = True
    results_folder_path = "mpc_test/test_results"

    # UNCOMMENT TO GENERATE CI CF PAIRS
    # _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)
    # generate_CI_CF_pairs(args, settings_file = None, results_folder_path = results_folder_path)

    args.controller_name = "SMDI" # Select controller
    
    # Old results: Results folder of the CI CF pairs
    call_simulation_with_CI_CF_pairs(args, results_folder_path)