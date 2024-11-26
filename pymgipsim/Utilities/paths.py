import os

"""
Path for the default settings
"""

simulator_path = os.path.abspath('.\\pymgipsim')

scenarios_path = os.path.abspath('.\\Scenarios')

models_path = os.path.abspath('.\\pymgipsim\\VirtualPatient\\Models')

controller_path = os.path.abspath('.\\pymgipsim\\Controllers')

default_settings_path = os.path.abspath('.\\pymgipsim\\Settings\\DefaultSettings')

results_path = os.path.abspath('.\\SimulationResults')

if __name__ == '__main__':

    print(simulator_path)

    print(default_settings_path)

    print(results_path)
