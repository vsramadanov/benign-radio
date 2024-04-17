from abc import ABC
from abc import abstractmethod
from numpy import array
from scipy.signal import lfilter

from simulation.unit import SimUnit


class ChannelBase(ABC, SimUnit):
    def __init__(self) -> None:
        super().__init__()

    def process(self, signal):
        pass_loss = self._pass_loss()
        channel_ir = self._impulse_responce()

        return pass_loss * lfilter(channel_ir, array([1]), signal, axis=0)

    @abstractmethod
    def _pass_loss(self):
        pass

    @abstractmethod
    def _impulse_responce(self):
        pass
