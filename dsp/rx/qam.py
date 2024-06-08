import numpy as np

from simulation.unit import SimUnit

from dsp.common.qam import QAMConstellation


class QAMSoftDemodulator(SimUnit):
    def __init__(self, constellation: QAMConstellation) -> None:
        self.constellation = constellation

        # suppose that constellation has rectangular layout
        symbols = constellation.symbols

        re_borders = np.zeros(symbols.shape[1], dtype=np.float64)
        im_borders = np.zeros(symbols.shape[0], dtype=np.float64)

        re_borders[0] = -np.Inf
        for k in range(1, symbols.shape[1]):
            re_borders[k] = np.mean(np.real(symbols[0, k-1:k+1]))

        im_borders[0] = -np.Inf
        for k in range(1, symbols.shape[0]):
            im_borders[k] = np.mean(np.imag(symbols[k-1:k+1, 0]))

        self.re_borders = re_borders
        self.im_borders = im_borders
        self.nbits = constellation.pow

    def process(self, symbols: np.array) -> np.array:

        re = np.real(symbols)
        im = np.imag(symbols)

        re_idx = np.searchsorted(self.re_borders, re) - 1
        im_idx = np.searchsorted(self.im_borders, im) - 1

        values = self.constellation.values[im_idx, re_idx]
        bits = np.unpackbits(values.astype(np.uint8)
                                   .reshape((-1, 1)),
                             axis=1,
                             bitorder='little',
                             )[:, :self.nbits].flatten()
        return bits
