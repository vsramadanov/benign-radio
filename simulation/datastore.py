from typing import Any
from utils import Singleton


class DaraStore(metaclass=Singleton):
    def __init__(self, names, callback=None) -> None:
        self.data = {}
        for name in names:
            self.data[name] = []

    def store(self, module, name, value):
        full_name = module + '.' + name
        entry = self.data.get(full_name, None)
        if entry:
            entry.append(value)
