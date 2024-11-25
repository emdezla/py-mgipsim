Inputs
========
.. _inputs target:


.. note::
	All inputs are defined by a series of (start time, magnitude and duration) triplets. Durations are not defined on the user side. For instance, the basal insulin rate is kept constant, for the bolus insulin a very short duration is automatically assigned.

	During definition the inputs have a fixed unit of measure and they will be converted to the unit of the selected model automatically.

Meal and snack
-------------------

Meals and snack can be defined by the user or by generating random intakes based on probability distributions...
Describe the carb <-> energy stuffs / the way the random generation occurs

**Meal:**

40 minutes of peak absorption is assumed for the meals.

- Used in: :ref:`T1DM Extended Hovorka <hovorka target>`, :ref:`T1DM IVP <ivp target>`

- Unit of measure: gramms

- Definition: Series of start times and magnitudes.


**Snack:**

20 minutes of peak absorption is assumed for the snacks.

- Used in: :ref:`T1DM Extended Hovorka <hovorka target>`

- Unit of measure: gramms

- Definition: Series of start times and magnitudes.


Insulin
-------------------

**Basal insulin:**

By default, basal rates are used that are defined in the demographic info field of the patient JSON file. Basal rates are kept constants until new rate is not defined.

- Used in: :ref:`T1DM Extended Hovorka <hovorka target>` and :ref:`T1DM IVP <ivp target>`

- Unit of measure: Units/hour

- Definition: Series of start times and magnitudes.


**Bolus insulin:**

By default, bolus values are calculated based on the carbohydrate-to-insulin ratio that are defined in the demographic info field of the patient JSON file. 

- Used in: :ref:`T1DM Extended Hovorka <hovorka target>` and :ref:`T1DM IVP <ivp target>`

- Unit of measure: Unit

- Definition: Series of start times and magnitudes.


SGL2i
-------------------


Activity
-------------------

.. note::
	Currently only the :ref:`T1DM Extended Hovorka <hovorka target>` model supports this feature. The input to the extended Hovorka model is heart rate, thus first the exerted power is translated to heart rate based on the demographics information of the patient.


**Running:**

Running activity is defined by the start time, duration, running speed [MPH] and incline [%]. Running speed and incline is translated to power based on the demographics information of the patient.

**Cycling:**

Cycling activity is defined by the start time, duration and average exerted power [watts].