import os
import logging
import pickle as pkl

from dataclasses import dataclass
from collections import defaultdict

from utils import Singleton


class StoreHandler():
    def __init__(self) -> None:
        self.impl = defaultdict(list)

    def store(self, **kwargs):
        for point, data in kwargs.items():
            self.impl[point].append(data)


class NoStoreHandler():
    def __init__(self) -> None:
        pass

    def store(self, **kwargs):
        pass


@dataclass(frozen=True)
class DataStoreConfig:
    path: str
    names: list[str]


class DaraStore(metaclass=Singleton):
    def __init__(self, config: DataStoreConfig) -> None:
        self.data = {}
        self.config = config
        self.logger = logging.getLogger('simulation.DataStore')
        self.logger.info('init DataStore')

        for name in config.names:
            self.data[name] = StoreHandler()
            self.logger.debug(f'set {name} class')

        main_store = self.data.get('__main__', None)
        self.store = main_store.store if main_store else NoStoreHandler()

    def flush(self):
        if not self.config:
            self.logger.info(f'nothing to write')
            return

        path = self.config.path
        os.makedirs(path, exist_ok=True)
        for name, storage in self.data.items():
            filename = os.path.join(self.config.path, name + '.pkl')
            with open(filename, 'wb') as fd:
                pkl.dump(storage.impl, fd)
            self.logger.info(
                f'succesfully write {filename} file, {storage.impl.keys()}')
