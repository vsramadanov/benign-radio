import numpy as np

from numpy.fft import fft
from numpy.fft import ifft

from simulation.unit import SimUnit


class FftUpsampler(SimUnit):

    def __init__(self, scale):
        self.scale = scale

    def process(self, x: np.array) -> np.array:
        X = fft(x).reshape((2, -1))
        Xu = np.concatenate(
            (
                X[0, :],
                np.zeros(len(x) * (self.scale - 1)),
                X[1, :]
            ),
            axis=0)

        if np.iscomplexobj(x):
            return self.scale * ifft(Xu)
        else:
            return self.scale * np.real(ifft(Xu))
