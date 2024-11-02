import numpy as np
from numpy.fft import fft

from ...common.ofdm_config import OFDMconfig
from ...common.ofdm_config import GItype
from simulation.unit import SimUnit


class OFDMfrontend(SimUnit):
    def __init__(self, config: OFDMconfig) -> None:
        super().__init__()

        self.config = config
        self.__rm_gi_protector = {
            GItype.ZP: self.__rm_prefix_protector,
            GItype.CP: self.__rm_prefix_protector,
            GItype.CS: self.__rm_suffix_protector,
        }[config.Type]

    def __rm_prefix_protector(self, ofdm_symbols: np.array) -> np.array:
        return ofdm_symbols[:, self.config.GI:]

    def __rm_suffix_protector(self, ofdm_symbols: np.array) -> np.array:
        return ofdm_symbols[:, :-self.config.GI]

    def process(self, ofdm_symbols: np.array) -> np.array:
        symbol_len = self.config.Nsc + self.config.GI
        L = len(ofdm_symbols) // symbol_len  # number of OFDM symbols
        ofdm_symbols.resize(L, symbol_len)

        ofdm_symbols = self.__rm_gi_protector(ofdm_symbols)
        symbols = fft(ofdm_symbols, axis=1)

        return symbols
