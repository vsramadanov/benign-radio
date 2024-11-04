import numpy as np
from numpy.fft import ifft

from ..common.ofdm_config import OFDMconfig
from simulation.unit import SimUnit


class OFDM(SimUnit):
    def __init__(self, config: OFDMconfig) -> None:
        super().__init__()

        self.config = config
        self.__add_gi_protector = {
            'zero_padding': self.__add_zero_padding,
            'cyclic_prefix': self.__add_cyclic_prefix,
            'cyclic_suffix': self.__add_cyclic_suffix
        }[config.guard_interval_type]

    def __add_cyclic_prefix(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            ofdm_symbols[:, -self.config.guard_interval_length:],
            ofdm_symbols,
        ), axis=1)

    def __add_cyclic_suffix(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            ofdm_symbols,
            ofdm_symbols[:, :self.config.guard_interval_length],
        ), axis=1)

    def __add_zero_padding(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            np.zeros((n, self.config.guard_interval_length)),
            ofdm_symbols,
        ), axis=1)

    def process(self, symbols: np.array) -> np.array:
        L = len(symbols) // self.config.Nsc  # number of OFDM symbols
        symbols = symbols.reshape(L, self.config.Nsc)
        ofdm_symbols = ifft(symbols, axis=1)

        return self.__add_gi_protector(ofdm_symbols).flatten()
