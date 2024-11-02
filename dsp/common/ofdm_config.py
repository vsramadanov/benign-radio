from dataclasses import dataclass
from enum import Enum


class GItype(Enum):
    ZP = 0  # Zero padding
    CP = 1  # Cyclic prefix
    CS = 2  # Cyclic suffix


@dataclass(frozen=True)
class OFDMconfig:
    Nsc: int  # subcarriers number
    GI: int  # guard interval
    Type: GItype
