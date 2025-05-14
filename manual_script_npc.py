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
    # args.patient_names = ["Patient_1", "Patient_2"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.running_speed = 0.0 # Turn off physical activity
    args.plot_patient = 0 # Plots patient glucose, intakes, heartrate
    args.random_seed = 0
    args.number_of_days = 7
    args.no_progress_bar = True
    args.no_print = True
    generate_new_meals = True # Set to True to generate new meals, False to use previous from csv files

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

    # Hardcode random start_time arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_start_times = [400, 410, 420, 430, 440, 450, 460, 470, 480, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460]
    hardcoded_lunch_start_times = [730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 720, 730, 740, 750, 760, 770, 780, 790, 800]
    hardcoded_dinner_start_times = [1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160]
    # Hardcode random duration arrays for breakfast, lunch, and dinner
    hardcoded_breakfast_durations = [10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5]
    hardcoded_lunch_durations = [15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10]
    hardcoded_dinner_durations = [20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15]

    # Load meal statistics from mpc_meal_stats.json
    with open('meal_tir_stats.json', 'r') as f:
        meal_tir_stats : dict = json.load(f)

    # Iterate over each patient in the settings_file
    for patient_idx in range(settings_file.patient.number_of_subjects):

        # Get the mean and std for the current patient from the JSON file
        mean_meals_per_day = list(meal_tir_stats.values())[patient_idx]['mean_carb_intake']
        std_meals_per_day = list(meal_tir_stats.values())[patient_idx]['std_carb_intake']

        # Generate random meal values for each day such that the mean and std match the JSON values
        daily_meals = norm.rvs(loc=mean_meals_per_day, scale=std_meals_per_day, size=args.number_of_days)

        # Ensure no negative meal values
        daily_meals = np.abs(daily_meals)

        # Save the generated meal_carb magnitudes into a CSV file
        csv_filename = f"patient_{patient_idx + 1}_meal_carb_magnitudes.csv"


        if generate_new_meals:
            file = open(csv_filename, mode='w', newline='')
            writer = csv.writer(file)
            writer.writerow(["Day", "Meal", "Carb Magnitude"])
        else:
            # Load the meal carb magnitudes from the CSV file
            file = open(csv_filename, mode='r')
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                day = int(row[0])
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

            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx] = hardcoded_breakfast_start_times[day_idx]
            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx + 1] = hardcoded_lunch_start_times[day_idx]
            settings_file.inputs.meal_carb.start_time[patient_idx][3 * day_idx + 2] = hardcoded_dinner_start_times[day_idx]

            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx] = hardcoded_breakfast_durations[day_idx]
            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx + 1] = hardcoded_lunch_durations[day_idx]
            settings_file.inputs.meal_carb.duration[patient_idx][3 * day_idx + 2] = hardcoded_dinner_durations[day_idx]
        print(f"Patient {patient_idx + 1} daily carbs \nmean: generated: {np.mean(daily_meals):.2f} g in study: {mean_meals_per_day:.2f} g, \nstd: generated: {np.std(daily_meals):.2f} g in study: {std_meals_per_day:.2f} g")
        # Close the CSV file
        file.close()

        
    TIR_matrix = np.zeros((len(CI_range), len(CF_range), settings_file.patient.number_of_subjects))
    for i in tqdm(CI_range):
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

            # print(f"\rCI: {i}, CF: {j}, AVG TIR: {np.mean(TIR):.2f}%")

    # Select CI, CF value pair for each patient
    selected_CI_CF_pairs = []
    for patient_idx in range(settings_file.patient.number_of_subjects):
        mean_tir = list(meal_tir_stats.values())[patient_idx]['mean_tir']

        # Find the CI, CF pair where TIR matches the mean_tir
        closest_match = None
        closest_diff = float('inf')
        for ci_idx, ci in enumerate(CI_range):
            for cf_idx, cf in enumerate(CF_range):
                tir_value = TIR_matrix[ci_idx, cf_idx, patient_idx]
                diff = abs(tir_value - mean_tir)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_match = (ci, cf)

        selected_CI_CF_pairs.append(closest_match)

    # Print the selected CI, CF pairs for each patient
    for patient_idx, (ci, cf) in enumerate(selected_CI_CF_pairs):
        print(f"Patient {patient_idx + 1}: Selected CI = {ci}, CF = {cf}, TIR from simulation = {TIR_matrix[ci - CI_range[0], cf - CF_range[0], patient_idx]:.2f}%, Mean TIR from study = {list(meal_tir_stats.values())[patient_idx]['mean_tir']:.2f}%")
    # Save the selected CI, CF pairs into a JSON file
    selected_CI_CF_dict = {
        list(meal_tir_stats.keys())[patient_idx]: {"Selected_CI": ci, "Selected_CF": cf}
        for patient_idx, (ci, cf) in enumerate(selected_CI_CF_pairs)
    }

    with open('selected_CI_CF_pairs.json', 'w') as json_file:
        json.dump(selected_CI_CF_dict, json_file, indent=4)

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
    # fig.canvas.manager.full_screen_toggle() # Fullscreen
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
    plt.savefig('TIR_surface_plot.png', dpi=300)
    # plt.show()