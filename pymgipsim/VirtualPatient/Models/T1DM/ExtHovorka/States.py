import numpy as np

class States:
    def __init__(self):
        self.S1: np.ndarray = np.array([], dtype=float)
        self.S2: np.ndarray = np.array([], dtype=float)
        self.I: np.ndarray = np.array([], dtype=float)
        self.x1: np.ndarray = np.array([], dtype=float)
        self.x2: np.ndarray  = np.array([], dtype=float)
        self.x3: np.ndarray  = np.array([], dtype=float)
        self.D1: np.ndarray  = np.array([], dtype=float)
        self.D2: np.ndarray  = np.array([], dtype=float)
        self.DH1: np.ndarray  = np.array([], dtype=float)
        self.DH2: np.ndarray  = np.array([], dtype=float)
        self.Efast: np.ndarray  = np.array([], dtype=float)
        self.Eslow: np.ndarray  = np.array([], dtype=float)
        self.Ehigh: np.ndarray  = np.array([], dtype=float)
        self.Q1: np.ndarray  = np.array([], dtype=float)
        self.Q2: np.ndarray  = np.array([], dtype=float)

        self.Gsub: np.ndarray  = np.array([], dtype=float)

        self.as_array: np.ndarray  = np.array([], dtype=float)

        self.state_names = ['S1', 'S2', 'I', 'x1', 'x2', 'x3', 'Q1', 'Q2', 'IG', 'D1Slow', 'D2Slow',
                                'D1Fast', 'D2Fast', 'EEfast', 'EEhighintensity', 'EElongeffect']

        self.state_units= ['mU/min', 'S2', 'I', 'x1', 'x2', 'x3', 'Q1', 'Q2', 'mmol/L', 'mmol/min', 'mmol/min',
                                'mmol/min', 'mmol/min', 'EEfast', 'EEhighintensity', 'EElongeffect']