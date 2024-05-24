from typing import Any
from utils import Singleton


class Storage():
    def __init__(self) -> None:
        self.impl = {}

    def store(self, **kwargs):
        for var, data in kwargs.items():
            self.impl.get(var, []).append(data)


class DaraStore(metaclass=Singleton):
    def __init__(self, names=[]) -> None:
        self.data = {}
        for name in names:
            self.data[name] = Storage()

    def save(self):
        pass


class NoStoreHandler():
    def __init__(self) -> None:
        pass

    def store(self, **kwargs):
        pass
