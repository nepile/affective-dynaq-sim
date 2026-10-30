.. affective-dynaq documentation master file

Affective Dyna-Q GridWorld Simulator Documentation
==================================================

Welcome to the **Affective Dyna-Q GridWorld Simulator** technical documentation.
This research framework implements reinforcement learning pathfinding agents
combined with the dimensional Valence-Arousal core affect model developed by
Johannes Feldmaier and Klaus Diepold (2014).

Project Architecture Overview
-----------------------------
* **Maze Generation & Validation**:
  Procedural grid environments generated using Randomized Kruskal's Algorithm and
  Cellular Automata. Path viability, start and goal placements are computed via
  Maximum Geodesic Distance in the largest connected component.

* **Environments**:
  Modular GridWorld environments supporting standard baseline Dyna-Q benchmarks,
  classic continuous dynamical frequency monitoring (Valence & Arousal), and
  artificial neural network (ANN) reward-shaping interfaces.

* **Affective Dynamics**:
  Real-time tracking of short-term memory frequencies:
  collision frequency :math:`f_{col}`, direction change frequency :math:`f_{dir}`,
  goal-in-sight visibility :math:`f_{goal}` (via Bresenham line raycasting), and
  distance-to-origin progression :math:`f_{right}`.

* **Agent & Policies**:
  Tabular Dyna-Q reinforcement learning integrating direct Q-learning, transition
  model recording, and simulated experience replay planning.

* **User Interfaces**:
  CLI terminal runner (headless) and interactive Pygame graphical viewer (GUI),
  featuring automatic publication-ready PNG figure exports per seed.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules
