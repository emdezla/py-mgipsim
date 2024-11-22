import numpy as np

class States:
    def __init__(self):
        self.G: np.ndarray = np.array([], dtype=float)
        self.M: np.ndarray = np.array([], dtype=float)
        self.vL: np.ndarray = np.array([], dtype=float)

        self.as_array: np.ndarray  = np.array([], dtype=float)

        self.state_names = ['G', 'M', 'vL']
        self.state_units = ['-', '-', '-']