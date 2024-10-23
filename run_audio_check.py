import numpy as np
import logging

import matplotlib.pyplot as plt

from simulation.params import SimParams
from simulation.datastore import DaraStore
from simulation.datastore import DataStoreConfig

from test.audio import AudioChannelConfig
from test.audio import RawAudioChannel

from scipy.signal import correlate
from scipy.signal import chirp

#
# Config
#

SimParams(
    fc=int(10e3),
    fs=int(20e3),
)

DaraStore(
    DataStoreConfig(
        names=[],
        path='',
    ))

RATE = 48000

# Configure Logger
logging.basicConfig(filename=f'out/sound_check.log', level=logging.INFO)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(level=logging.CRITICAL)

#
# Prepare data
#

if __name__ == "__main__":

    fc = 440  # Hz
    Tmax = 1  # 1 sec
    t_sig = np.arange(Tmax * RATE) / RATE
    signal = np.sin(2 * np.pi * fc * t_sig)
    signal_q = (signal * (2 ** 15 - 1)).astype(np.int16)

    with RawAudioChannel(
            config=AudioChannelConfig(
                rate=RATE,
                channels=1,
                format='',
                device="default",
            )) as ch:
        input_q = ch.route_audio(signal_q)

    t_in = np.arange(len(input_q)) / RATE

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_sig, signal_q)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t_in, input_q)
    plt.show()
