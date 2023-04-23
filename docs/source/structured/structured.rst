Structured
==========

The `structured` module is intended for working with tabular data.
It currently only supports data in a `pandas` DataFrame.

There are 3 components to the `structured` library:
Manipulators, Challengers, and Explainers.
 * Manipulators take in raw data and allow the user to apply attacks.
 * Challengers encapsulate a set of Manipulators. They train models on a set of
   predefined challenges, or sequences of manipulations.
 * Explainers apply an explainability method to (trained model, dataset) pairs.
   They can also be used to evaluate all of the challenges in a Challenger.

The `core` module contains the StructuredManipulator class. It also contains
base classes for implementing Challengers and Explainers.

The `samples` module contains implementations of challengers and explainers.

.. toctree::
    :maxdepth: 1

    api
