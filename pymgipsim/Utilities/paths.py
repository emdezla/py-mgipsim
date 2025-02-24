import os

"""
Path for the default settings
"""

simulator_path = os.path.abspath(os.path.join('.', 'pymgipsim'))

scenarios_path = os.path.abspath(os.path.join('.', 'Scenarios'))

models_path = os.path.abspath(os.path.join('.', 'pymgipsim', 'VirtualPatient', 'Models'))

controller_path = os.path.abspath(os.path.join('.', 'pymgipsim', 'Controllers'))

default_settings_path = os.path.abspath(os.path.join('.', 'pymgipsim', 'Settings', 'DefaultSettings'))

results_path = os.path.abspath(os.path.join('.', 'SimulationResults'))

if __name__ == '__main__':
    print(simulator_path)
    print(default_settings_path)
    print(results_path)
