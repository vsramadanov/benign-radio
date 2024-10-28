import logging
import argparse
import importlib
import yaml

from simulation.params import SimParams
from simulation.datastore import DataStore
from simulation.datastore import DataStoreConfig

parser = argparse.ArgumentParser(description='update YAML configuration.')
parser.add_argument('-c', '--config', type=str,
                    help='path to the YAML initial config file.')
parser.add_argument('--set', action='append', nargs=2, metavar=('key', 'value'),
                    help='Set config parameter. Example: --set param1.name value1')
parser.add_argument('scenario', type=str, help='a scenario to run')


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def save_config(file_path, config):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)


if __name__ == '__main__':

    args = parser.parse_args()
    config = load_config(args.config)

    if args.set:
        for key, value in args.set:
            field = config
            for entry in key.split('.'):
                field = field[entry]
            field = value

    logging.basicConfig(**config['logging'])
    for module in ['matplotlib', 'PIL']:
        logger = logging.getLogger(module)
        logger.setLevel(level=logging.CRITICAL)

    SimParams(**config['SimParams'])

    dsconfig = DataStoreConfig(**config['DataStore'])
    ds = DataStore(config=dsconfig)

    scenario = importlib.import_module(args.scenario).Scenario()

    try:
        scenario.run(config)

    except Exception as e:
        logging.critical(f"Oops: {type(e).__name__}: {e}. Simulation stopped")

    finally:
        ds.flush()
