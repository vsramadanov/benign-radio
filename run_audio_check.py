import numpy as np
import logging

import matplotlib.pyplot as plt

from simulation.params import SimParams

from test.audio import AudioChannelConfig
from test.audio import AudioChannel

from scipy.signal import correlate
from scipy.signal import chirp

#
# Config
#

SimParams(
    fc=int(10e3),
    fs=int(20e3),
)

RATE = 48000

# Configure Logger
logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(level=logging.CRITICAL)

#
# Prepare data
#

t0 = 0
t1 = 5
f0 = 10
f1 = 20e3
time = np.linspace(t0, t1, (t1 - t0) * RATE)
signal = chirp(t=time, t1=t1, f0=f0, f1=f1, phi=270, method='linear')

#
# Transmit & Receive over the audio devices
#

with AudioChannel(
        config=AudioChannelConfig(
            fs=RATE,
            channels=1,
            format='Int16',
            chunk=4096
        )) as channel:

    recv = channel.process_raw(tx_signal=(2**14 * signal).astype(np.int16))

#
# Process received data
#

trecv = np.arange(recv.shape[0]) / RATE

# find precision chirp location
cref = chirp(t=time, f0=f0, f1=f1, t1=5, phi=270) + \
    1j*chirp(t=time, f0=f0, f1=f1, t1=5, phi=0)
Y = np.abs(correlate(recv, cref, mode='valid'))
idx = np.argmax(Y)

logging.debug(f"CHIRP signal found at {idx} sample")

# Spectrum
xc = recv[idx:idx + (t1 - t0) * RATE]  # crop recv signal

freq_resp = np.abs(np.fft.fft(xc))
freqs = RATE * np.arange(freq_resp.shape[0]) / freq_resp.shape[0]
Len = freqs.shape[0] // 2

fig, ax = plt.subplots(1, 2)
ax[0].plot(trecv, recv)
ax[0].set_xlabel('time, sec')
ax[0].set_ylabel('Signal')
ax[0].grid(True)

ax[1].plot(freqs[:Len], 10*np.log10(freq_resp[:Len] / np.max(freq_resp[:Len])))
ax[1].set_xlabel('freq, Hz')
ax[1].set_ylabel('Responce')
ax[1].grid(True)

plt.show()
