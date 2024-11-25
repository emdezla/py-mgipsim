:tocdepth: 1

Quickstart
============

To get started immediately, run :code:`interface_cmd.py` to activate a command line interface (CLI).


Run the simulator
------------
* :code:`simulate` This command is used to run the simulation with the assigned settings.
* Using just :code:`simulate` generates a default scenario (`defaults </_autosummary/pymgipsim.Settings.DefaultSettings.html>`_).
* Predefined scenarios are available in the :code:`./Scenarios/` folder. To use one of them, use :code:`load -sn <scenario name>` and :code:`simulate`. 
* Results are stored in the directory :code:`./SimulationResults/<results folder with datetime from simulation run time>`

Plot the results
------------
* :code:`plot` This command is used to generate plots from the simulation results.
* :code:`plot -h` lists the arguments that can be used with :code:`plot <arguments>`
* :code:`plot` with no arguments generates and saves all figures without displaying them.
* :code:`plot --all` both generates and displays all figures.