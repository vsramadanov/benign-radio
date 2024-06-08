import numpy as np


class QAMConstellation:
    def __init__(self, order) -> None:

        assert ((order & (order - 1)) == 0)
        pow = int(np.round(np.log2(order)))

        self.order = order
        self.pow = pow

        ytiks = 2 ** int(np.floor(pow / 2))
        xtiks = order // ytiks
        x = np.linspace(-1, 1, xtiks)
        y = np.linspace(-1, 1, ytiks) if ytiks > 1 else 0

        re, im = np.meshgrid(x, y)
        self.__symbols = re + 1j * im
        self.__values = np.arange(order).reshape(self.__symbols.shape)

    @property
    def symbols(self):
        return self.__symbols

    @property
    def values(self):
        return self.__values

    @property
    def mapping(self):
        return zip(
            self.__symbols.flatten(),
            self.__values.flatten(),
        )
