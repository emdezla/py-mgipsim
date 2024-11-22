import numpy as np

class States:
    def __init__(self):
        self.BW: np.ndarray = np.array([], dtype=float)
        self.as_array: np.ndarray  = np.array([], dtype=float)

        self.state_names = ['BW']
        self.state_units = ['kg']