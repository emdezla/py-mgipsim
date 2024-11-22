import numpy as np
from numba import jit_module

jit_module(nopython = True)

"""
Unit Conversions
"""

class TimeUnits:

    @staticmethod
    def convert_hour_to_min(hours):
        return hours * 60

    @staticmethod
    def convert_inverse_hour_inverse_minute(inverse_hour):
        """ 1/h to 1/min"""
        return inverse_hour / 60

    @staticmethod
    def convert_minutes_to_days(minutes):
        return minutes / 60 / 24
    
    @staticmethod
    def calculate_time_adjustment_array(dimension):

        time_adjustment_array = np.ones((dimension)).astype(float)

        for i in range(dimension[-1]):

            time_adjustment_array[:, i] *= UnitConversion.time.convert_hour_to_min(24*i)

        return time_adjustment_array.astype(float)

    @staticmethod
    def convert_days_to_minutes(days):
        return days * 24 * 60

class MetricUnits:

    @staticmethod
    def base_to_milli(unit):
        return unit * 1e3

    @staticmethod
    def milli_to_base(unit):
        return unit/1e3
    
class InsulinUnits:

    @staticmethod
    def Uhr_to_uUmin(Uhr):
        return Uhr/ 60.0 * 1E6

    @staticmethod
    def uUmin_to_Uhr(uUmin):
        return uUmin* 60.0 / 1E6

    @staticmethod
    def Uhr_to_mUmin(Uhr):
        return Uhr*1000/60

    @staticmethod
    def mUmin_to_Uhr(mUmin):
        return mUmin/1000*60

    @staticmethod
    def U_to_mU(U):
        return U*1000

    @staticmethod
    def U_to_uU(U):
        return U * 1e6
    

class GlucoseUnits:

    @staticmethod
    def g_glucose_to_mol(g_glucose):
        return (g_glucose / 180.156)

    @staticmethod
    def g_glucose_to_mmol(g_glucose):
        return (g_glucose / 180.156) * 1000

    @staticmethod
    def mmol_glcuose_to_g(mmol_glucose):
        return (mmol_glucose / 1000) * 180.156

    @staticmethod
    def concentration_mmolL_to_mgdL(mmolL):
        return mmolL*18

    @staticmethod
    def concentration_mgdl_to_mmolL(mgdL):
        return mgdL/18

    @staticmethod
    def energy_g_glucose_to_kkcal(g):
        return g*4

    @staticmethod
    def energy_kkcal_to_g_glucose_equiv(kcal):
        return kcal/4


class UnitConversion:

    time = TimeUnits()

    metric = MetricUnits()

    insulin = InsulinUnits()

    glucose = GlucoseUnits()


"""
Units
"""

GLUCOSE_KCAL_PER_GRAM = 4 # kcal/g

"""
Constants
"""

# Define random seed
DEFAULT_RANDOM_SEED = 402