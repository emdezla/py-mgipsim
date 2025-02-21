from pymgipsim.Utilities.Scenario import demographic_info
from .NMPC import NMPC
from pymgipsim.Utilities.units_conversions_constants import UnitConversion

class Controller:
    name = "SMDI"
    def __init__(self, scenario_instance):
        self.nmpc = NMPC(scenario_instance)

    def run(self, measurements, inputs, states, sample):
        self.nmpc.run(sample, UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[0]))
        return