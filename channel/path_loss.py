import numpy as np

from .channel_base import ChannelBase


class PathLossChannel(ChannelBase):

    def __init__(self, R) -> None:
        super().__init__()

        self.range = R

        lam = 3e8 / self.params.fc
        delta = R - np.floor(R / lam) * lam
        phase = np.exp(-1j * 2 * np.pi * delta / lam)
        self.phase = np.array([phase])

    def _pass_loss(self):
        return 1 / self.range ** 2

    def _impulse_responce(self):
        return self.phase
