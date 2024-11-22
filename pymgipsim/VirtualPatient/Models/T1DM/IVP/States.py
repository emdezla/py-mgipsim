import numpy as np

class States:
    def __init__(self):
        self.G: np.ndarray = np.array([], dtype=float)
        self.Ieff: np.ndarray = np.array([], dtype=float)
        self.Ip: np.ndarray = np.array([], dtype=float)
        self.Isc: np.ndarray = np.array([], dtype=float)

        self.as_array: np.ndarray  = np.array([], dtype=float)

        self.state_names = ['G', 'Ieff', 'Ip', 'Isc']
        self.state_units = ['mg/dL', '1/min', 'uU/mL', 'uU/mL']