import pyaudio
import numpy as np
from scipy.signal import chirp
from scipy.signal import correlate

import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

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


class CallbackPlayer:
    def __init__(self, src: np.array) -> None:
        self.data = src.astype(np.int16).tobytes()
        self.len = len(src)
        self.frame = 0

    def __call__(self, in_data, frame_count, time_info, status):

        begin = self.frame
        end = min(self.frame + frame_count, self.len)

        self.frame = end

        data = self.data[2*begin:2*end]  # int16 contains 2 bytes
        if (end < self.len):
            return data, pyaudio.paContinue

        return data, pyaudio.paComplete


class CallbackRecorder:
    def __init__(self, dst: list) -> None:
        self.data = dst

    def __call__(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        self.data.append(data)
        return None, pyaudio.paContinue


print("Open PyAudio")

p = pyaudio.PyAudio()
stream_play = p.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     output=True,
                     stream_callback=CallbackPlayer(src=2**14 * signal))

records = []
stream_record = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       stream_callback=CallbackRecorder(dst=records))

stream_record.start_stream()
stream_play.start_stream()

while stream_play.is_active() and stream_record.is_active():
    pass

print("Passed spinlock")

stream_play.stop_stream()
stream_play.close()

print("stream_play closed")

stream_record.stop_stream()
stream_record.close()

print("stream_record closed")

p.terminate()

print("PyAudio terminated")

#
# Process received data
#

recv = np.concatenate(records, axis=0)
trecv = np.arange(recv.shape[0]) / RATE

print(f"Received {recv.shape[0]} samples, "
      f"{recv.shape[0] / RATE} seconds of data")

# find precision chirp location
cref = chirp(t=time, f0=f0, f1=f1, t1=5, phi=270) + \
    1j*chirp(t=time, f0=f0, f1=f1, t1=5, phi=0)
Y = np.abs(correlate(recv, cref, mode='valid'))
idx = np.argmax(Y)
print(f"CHIRP signal found at {idx} sample")

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
