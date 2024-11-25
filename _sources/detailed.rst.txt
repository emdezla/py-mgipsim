For the curious ones
=====================
.. _detailed target:


Here we aim to give a little more detailed information how the simulator workflow is layed out, and what are the underlying inputs, models and methods that define the observed simulation results.


.. toctree::

	inputs
	models

A scenario is a unique descriptor of a simulation. Outside of the simulator a scenario is described in a JSON file. At the end of each simulation the scenario is exported and can be reused later to make a simulation reproducable. The core element of the simulator is the scenario JSON file/class. Between the class and the file there is a one-to-one mapping, all the fields in the file are mirrored as the scenario class attributes. For the detailed description of the fields in the scenario please refer to the `docs </_autosummary/pymgipsim.Utilities.Scenario.scenario.html>`_.

By using the command line interface, the user apply modifications to a default scenario instance (loaded from the default JSON file) (`defaults </_autosummary/pymgipsim.Settings.DefaultSettings.html>`_). Until the simulation command is not called, only the scenario instance is manipulated and saved in each module. The CLI parses inputs according to the library [argparse](https://docs.python.org/3/library/argparse.html).

.. image:: imgs/workflow.svg
   :target: _images/workflow.svg


**General settings (generate_settings.py)**

:code:`>> settings` 

.. automodule:: generate_settings
	:exclude-members: generate_simulation_settings_main

**Virtual cohort settings (generate_subjects.py)**

:code:`>> cohort` 

.. automodule:: generate_subjects
	:exclude-members: generate_virtual_subjects_main

**Input settings (generate_inputs.py)**

:code:`>> inputs` 

.. automodule:: generate_inputs

**Simulator (generate_results.py)**

:code:`>> simulate` 

.. automodule:: generate_results

**Plotting (generate_plots.py**

:code:`>> plot` 

.. automodule:: generate_plots
