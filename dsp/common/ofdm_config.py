from dataclasses import dataclass


@dataclass(frozen=True)
class OFDMconfig:
    Nsc: int  # subcarriers number
    guard_interval_length: int
    guard_interval_type: str
