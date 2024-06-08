import logging

from .params import SimParams
from .datastore import DaraStore
from .datastore import NoStoreHandler


class SimUnit:
    def __init__(self) -> None:
        self.params = SimParams()
        storage = DaraStore().data.get(self.full_cls_name, None)
        if storage:
            self.store = storage.store
            self.logger.info(f"add DataStore entry")

        else:
            self.store = NoStoreHandler.store
            self.logger.info(f"is ignored by DataStore")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.full_cls_name = cls.__module__ + '.' + cls.__name__
        cls.logger = logging.getLogger(cls.full_cls_name)

        cls.__str__ = lambda self: f"{cls.__name__}, config={self.config}"
