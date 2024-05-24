import numpy as np

from simulation.unit import SimUnit


class OFDMChannleEqualizer(SimUnit):
    def __init__(self) -> None:
        self.config = []

    def process(self, symbols: np.array, chest: np.array) -> np.array:
        return symbols / chest
