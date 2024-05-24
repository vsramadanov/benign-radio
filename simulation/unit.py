import logging

from .params import SimParams
from .datastore import DaraStore
from .datastore import NoStoreHandler


class SimUnit:
    def __init__(self) -> None:
        self.params = SimParams()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        full_cls_name = cls.__module__ + '.' + cls.__name__
        cls.logger = logging.getLogger(full_cls_name)

        storage = DaraStore().data.get(full_cls_name, None)
        if storage:
            cls.store = storage.store

        else:
            cls.store = NoStoreHandler.store

        cls.__str__ = lambda self: f"{cls.__name__}, config={self.config}"
