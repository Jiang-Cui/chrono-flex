implicit-beams-gpu
====
This is a general purpose simulator for three dimensional flexible multibody dynamics problems. This implementation uses gradient-deficient Absolute Nodal Coordinate Formulation (ANCF) beam elements to model slender beams. These are two node elements with one position vector and only one gradient vector used as nodal coordinates. Each node thus has six coordinates: three components of the global position vector and three components of the position vector gradient at the node. This formulation displays no shear locking problems for thin and stiff beams. The gradient-deficient ANCF beam element does not describe rotation of the beam about its own axis so the torsional effects cannot be modeled.

Features
----
This software provides a suite of flexible body suppo, including:
- gradient-deficient beam elements
- the ability to connect these elements with bilateral constraints
- multiple solvers, including [Spike::GPU](http://spikegpu.sbel.org)
- contact with friction

Example
----
asdlfkjsdaf

Install
----
asdflkajsdf

Credits
----
(c) Simulation-Based Engineering Laboratory, 2013
