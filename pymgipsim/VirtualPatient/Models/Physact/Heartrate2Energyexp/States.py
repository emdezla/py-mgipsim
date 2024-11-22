import numpy as np

class States:
    def __init__(self):
        self.EE: np.ndarray = np.array([], dtype=float)

        self.as_array: np.ndarray  = np.array([], dtype=float)

        self.state_names = ['EE']
        self.state_units = ['-']