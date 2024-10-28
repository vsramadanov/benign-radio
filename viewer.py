import argparse
import pickle as pkl
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing


def action_plot(name, data):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(np.real(data[0]))
    axs[0].plot(np.imag(data[0]))
    axs[0].title.set_text(name)

    axs[1].plot(np.abs(data[0]))

    plt.show()


def action_print(name, data):
    print(f'name: {name}, val: {data}')


def action_eye(name, data):
    plt.plot(np.real(data[0]), np.imag(data[0]), linestyle='', marker='o')
    plt.title(f'{name} eye diagram')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axis('equal')
    plt.show()


def apply_actions(actions: list, name, data):
    for action in actions:
        try:
            action(name, data)
            return
        except KeyboardInterrupt:
            return
        except:
            continue
    raise ValueError(f"no suitable action for entry: {name}!")


action_dict = {
    'plot': action_plot,
    'print': action_print,
    'eye': action_eye,
}

parser = argparse.ArgumentParser(description='viewer for datastore artifats')

parser.add_argument('filename', type=str, help='pkl file with stored data')
parser.add_argument('--list', action='store_true',
                    default=False, help='list all entries')
parser.add_argument('-f', '--filter', type=str, nargs='+',
                    help='filter entries to view')
parser.add_argument('-a', '--action', type=str, nargs='+', default=['plot', 'print'],
                    help='action to do with entry, first suitable will be applied')


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.filename, 'rb') as fd:
        dumps = pkl.load(fd, fix_imports=True, errors='strict', buffers=None)

    if args.list:
        for name in dumps.keys():
            print(name)
        exit(0)

    if args.filter is not None:
        dumps = {key: dumps[key] for key in args.filter}
        print("filter applied!")

    for name, val in dumps.items():
        print(f'considered {name} dump, len: {len(val)}')
        for idx, entry in enumerate(val):
            print(f'\t{name} #{idx}: shape {entry.shape}')

    actions = [action_dict[name] for name in args.action]
    view_dump = partial(apply_actions, actions)

    with multiprocessing.Pool(processes=len(dumps)) as p:
        p.starmap(view_dump, dumps.items())
