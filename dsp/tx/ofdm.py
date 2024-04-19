import numpy as np
from numpy.fft import ifft

from ..common.ofdm_params import OFDMparams
from ..common.ofdm_params import GItype
from simulation.unit import SimUnit


class OFDM(SimUnit):
    def __init__(self, params: OFDMparams) -> None:
        super().__init__()

        self.params = params
        self.__add_gi_protector = {
            GItype.ZP: self.__add_zero_pading,
            GItype.CP: self.__add_cyclic_prefix,
            GItype.CS: self.__add_cyclic_suffix
        }[params.Type]

    def __add_cyclic_prefix(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            ofdm_symbols[:, -self.params.GI:],
            ofdm_symbols,
        ), axis=1)

    def __add_cyclic_suffix(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            ofdm_symbols,
            ofdm_symbols[:, :self.params.GI],
        ), axis=1)

    def __add_zero_pading(self, ofdm_symbols: np.array) -> np.array:
        n, _ = ofdm_symbols.shape
        return np.concatenate((
            np.zeros((n, self.params.GI)),
            ofdm_symbols,
        ), axis=1)

    def process(self, symbols: np.array) -> np.array:
        L = len(symbols) // self.params.Ncs  # number of OFDM symbols
        symbols.resize(L, self.params.Ncs)
        ofdm_symbols = ifft(symbols, axis=1)

        return self.__add_gi_protector(ofdm_symbols).flatten()
