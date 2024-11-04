import numpy as np
from numpy.fft import fft

from ...common.ofdm_config import OFDMconfig
from simulation.unit import SimUnit


class OFDMfrontend(SimUnit):
    def __init__(self, config: OFDMconfig) -> None:
        super().__init__()

        self.config = config
        self.__rm_gi_protector = {
            'zero_padding': self.__rm_prefix_protector,
            'cyclic_prefix': self.__rm_prefix_protector,
            'cyclic_suffix': self.__rm_suffix_protector,
        }[config.guard_interval_type]

    def __rm_prefix_protector(self, ofdm_symbols: np.array) -> np.array:
        return ofdm_symbols[:, self.config.guard_interval_length:]

    def __rm_suffix_protector(self, ofdm_symbols: np.array) -> np.array:
        return ofdm_symbols[:, :-self.config.guard_interval_length]

    def process(self, ofdm_signal: np.array) -> np.array:
        symbol_len = self.config.Nsc + self.config.guard_interval_length
        Nsymb = len(ofdm_signal) // symbol_len  # number of OFDM symbols
        ofdm_symbols = np.resize(ofdm_signal, (Nsymb, symbol_len))

        ofdm_symbols = self.__rm_gi_protector(ofdm_symbols)
        symbols = fft(ofdm_symbols, axis=1)

        return symbols
