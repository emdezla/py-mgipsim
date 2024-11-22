from ..Utilities.units_conversions_constants import DEFAULT_RANDOM_SEED

import numpy as np
import sys
from datetime import datetime

np.random.seed(DEFAULT_RANDOM_SEED)

class SimulationMetaData:
    @staticmethod
    def generate_timestamp() -> str:
        """
        Generate a timestamp in the format "%m_%d_%Y_%H_%M_%S".

        Returns:
        - str
            Timestamp string.
        """
        return str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    @staticmethod
    def generate_system_information() -> str:
        """
        Generate system information including the Python version.

        Returns:
        - str
            System information.
        """
        return sys.version


