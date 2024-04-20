import numpy as np

from simulation.unit import SimUnit


class OFDMChannleEstimator(SimUnit):
    def __init__(self) -> None:
        pass

    def process(self, symbols: np.array) -> np.array:
        _, Nsc = symbols.shape
        return np.ones(Nsc)
