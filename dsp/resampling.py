import numpy as np

from dataclasses import dataclass
from fractions import Fraction

from scipy.signal import resample_poly

from simulation.unit import SimUnit


@dataclass
class PolyResamplerConfig:
    fin: int
    fout: int


class PolyResampler(SimUnit):
    def __init__(self, config: PolyResamplerConfig) -> None:
        super().__init__()

        self.config = config
        ratio = Fraction(
            numerator=int(config.fout),
            denominator=int(config.fin),
        )
        self.up = ratio.numerator
        self.down = ratio.denominator

        self.logger.info(f'resampling {self.up}/{self.down}')

    def process(self, input) -> np.array:
        return resample_poly(input, up=self.up, down=self.down)
