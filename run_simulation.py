import numpy as np

from simulation.params import SimParams
from channel.path_loss import PathLossChannel

SimParams(
    fc=10e9,
    fs=25e6,
)

channel = PathLossChannel(R=100)
channel.process(np.ones(10))
