import numpy as np

from simulation.unit import SimUnit

from .frontend import OFDMfrontend
from .equalizer import OFDMChannleEqualizer
from .estimator import OFDMChannleEstimator


class OFDMRxChain(SimUnit):
    def __init__(self,
                 frontend: OFDMfrontend,
                 estimator: OFDMChannleEstimator,
                 equalizer: OFDMChannleEqualizer) -> None:
        self.frontend = frontend
        self.estimator = estimator
        self.equalizer = equalizer

    def process(self, ofdm_symbols: np.array) -> np.array:
        symbols = self.frontend.process(ofdm_symbols)
        chest = self.estimator.process(symbols)

        eqsymbols = self.equalizer.process(symbols=symbols, chest=chest)

        return eqsymbols.flatten()
