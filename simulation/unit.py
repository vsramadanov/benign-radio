import logging

from .params import SimParams


class SimUnit:
    def __init__(self) -> None:
        self.params = SimParams()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.logger = logging.getLogger(cls.__module__ + '.' + cls.__name__)
