from abc import ABC, abstractmethod
from ..ODESolvers.ode_solvers import euler_single_step, rk4_single_step
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from tqdm import tqdm

class BaseSolver(ABC):

    def __init__(self, scenario_instance: scenario, model: BaseModel):
        # Directory where the results should be stored
        self.scenario_instance = scenario_instance

        self.model = model

        # ODE solver function
        self.set_solver(self.scenario_instance.settings.solver_name)


    def set_solver(self, solver_name):
        match solver_name:
            case "RK4":
                self.ode_solver = rk4_single_step
            case 'Euler':
                self.ode_solver = euler_single_step

    def do_simulation(self, no_progress_bar):
        """ Initialize """
        state_results = self.model.states.as_array
        inputs = self.model.inputs.as_array
        parameters = self.model.parameters.as_array

        state_results[:, :, 0] = self.model.initial_conditions.as_array
        for sample in tqdm(range(1, inputs.shape[2]), disable=no_progress_bar):

            state_results[:, :, sample] = self.ode_solver(
                f=self.model.model,
                time=float(sample),
                h=float(self.model.sampling_time),
                initial=state_results[:, :, sample - 1].copy(),
                parameters=parameters,
                inputs=inputs[:, :, sample - 1]
            )

        self.model.states.as_array = state_results
        self.model.inputs.as_array = inputs

        return state_results
