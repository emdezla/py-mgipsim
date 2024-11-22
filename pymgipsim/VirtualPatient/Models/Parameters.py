import numpy as np


class BaseParameters:

    def __init__(self):
        self.as_array: np.ndarray = np.array([], dtype=float)

    def export(self):
        export = {}
        export["model_parameters"] = self.as_array.tolist()
        return export

    def fromJSON(self, dict):
        """ Maps dictionary elements coming (from patient JSON files) to instance attributes.
        """

        for key, value in dict.items():
            setattr(self, key, np.asarray([value]))