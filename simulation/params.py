from dataclasses import dataclass

from utils import Singleton


@dataclass(frozen=True)
class SimParams(metaclass=Singleton):
    fc: int
    fs: int
