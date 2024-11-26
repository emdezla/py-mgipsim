from abc import ABC, abstractmethod
class BaseModel(ABC):
    """ Abstract Model class.

    Attributes:
        inputs
        states
        initial_conditions
        parameters
        time: ? Technically should be an attribute of the ModelSolver if possible. It is not model specific.

    """


    output_state: int = NotImplemented

    def __init__(self):
        self.states = NotImplemented
        self.inputs = NotImplemented
        self.parameters = NotImplemented
        self.states = NotImplemented
        self.time = NotImplemented
        self.sampling_time = NotImplemented
        self.initial_conditions = NotImplemented

    @abstractmethod
    def preprocessing(self):
        pass

    @staticmethod
    @abstractmethod
    def model(states, time, parameters, inputs):
        pass

    @staticmethod
    def rate_equations(states, time, parameters, inputs):
        pass

    @staticmethod
    @abstractmethod
    def output_equilibrium(parameters, inputs):
        pass
