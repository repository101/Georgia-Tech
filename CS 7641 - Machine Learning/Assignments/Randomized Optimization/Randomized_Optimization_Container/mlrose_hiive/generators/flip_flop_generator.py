""" Classes for defining optimization problem objects."""

# Author: Andrew Rollings
# License: BSD 3 clause

import numpy as np

from mlrose_hiive import FlipFlopOpt


class FlipFlopGenerator:
    @staticmethod
    def generate(seed, size=20, maximize=True):
        np.random.seed(seed)
        problem = FlipFlopOpt(length=size, maximize=maximize)
        return problem
