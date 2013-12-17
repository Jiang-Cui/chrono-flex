implicit-beams-gpu
====
This is a general purpose simulator for three dimensional flexible multibody dynamics problems. This implementation uses gradient-deficient Absolute Nodal Coordinate Formulation (ANCF) beam elements to model slender beams. These are two node elements with one position vector and only one gradient vector used as nodal coordinates. Each node thus has six coordinates: three components of the global position vector and three components of the position vector gradient at the node. This formulation displays no shear locking problems for thin and stiff beams. The gradient-deficient ANCF beam element does not describe rotation of the beam about its own axis so the torsional effects cannot be modeled.

![pic1691](https://f.cloud.github.com/assets/5438923/1767518/6158fad6-6755-11e3-9bf5-508598192e24.png)

Features
----
This software provides a suite of flexible body support implemented in parallel on the GPU, including:
* gradient-deficient beam elements
* the ability to connect these elements with bilateral constraints
* multiple solvers, including [Spike::GPU](http://spikegpu.sbel.org)
* contact with friction

Example
----
```c
// create the ANCF system
ANCFSystem sys;
sys.setTimeStep(1e-3);
sys.setTolerance(1e-6);

// create an element and add it to the system
double length = 1;
double r = 0.02;
double E = 2e7;
double rho = 2200;
double nu = .3;
Element element = Element(Node(0, 0, 0, 1, 0, 0), Node(length, 0, 0, 1, 0, 0), r, nu, E, rho);
sys.addElement(&element);

// pin the first node to the ground
sys.addConstraint_AbsoluteSpherical(0);

sys.initializeSystem();

// perform a single time step
sys.DoTimeStep();
```

Animations
----
* [Net Catch](https://vimeo.com/62115217)
* [Net Swing](https://vimeo.com/52520542)
* [Beam Cage](https://vimeo.com/58188595)
* [Worms](https://vimeo.com/58189498)

Install
----
* Download and install [CUDA](https://developer.nvidia.com/cuda-downloads) 
* Clone this repository
* Use [CMake](http://www.cmake.org) to generate a native makefile and workspace that can be used in the compiler environment of your choice

Literature
----
* Melanz, D., Khude, N., Jayakumar, P., & Negrut, D. (2013). A Matrix-Free Newton-Krylov Parallel Implicit Implementation of the Absolute Nodal Coordinate Formulation. Journal of Computational and Nonlinear Dynamics, 8(4).
* Melanz, D. (2012). [On the validation and applications of a parallel flexible multi-body dynamics implementation](http://sbel.wisc.edu/documents/DanielMelanzMSthesisFinal.pdf). University of Wisconsin - Madison.
* Khude, N., Stanciulescu, I., Melanz, D., & Negrut, D. (2013). [Efficient Parallel Simulation of Large Flexible Body Systems With Multiple Contacts](http://computationalnonlinear.asmedigitalcollection.asme.org/article.aspx?articleid=1673047). Journal of Computational and Nonlinear Dynamics, 8(4).
* Mazhar, H., Heyn, T., Pazouki, A., Melanz, D., Seidl, A., Bartholomew, A., Negrut, D. (2013). [Chrono: a parallel multi-physics library for rigid-body, flexible-body, and fluid dynamics](http://www.mech-sci.net/4/49/2013/ms-4-49-2013.html). Mechanical Sciences, 4(1), 49–64.

Credits
----
(c) Simulation-Based Engineering Laboratory, 2013
