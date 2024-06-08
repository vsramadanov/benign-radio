import numpy as np

from simulation.unit import SimUnit

from dsp.common.qam import QAMConstellation


class QAMModulator(SimUnit):
    def __init__(self, constellation: QAMConstellation) -> None:
        self.constellation = constellation

        table = np.zeros(constellation.order, dtype=np.complex128)
        for symb, val in constellation.mapping:
            table[val] = symb
        self.table = table
        self.nbits = constellation.pow

    def process(self, data: np.array) -> np.array:
        idxs = np.packbits(data.reshape((-1, self.nbits)),
                           axis=1, bitorder='little').flatten()

        return self.table[idxs]
