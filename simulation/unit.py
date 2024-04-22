import logging

from .params import SimParams


class SimUnit:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.params = SimParams()
        cls.logger = logging.getLogger(cls.__module__ + '.' + cls.__name__)
