import numpy as np
import logging

from scipy.signal import lfilter
from scipy.signal import correlate

import matplotlib.pyplot as plt

from simulation.params import SimParams
from simulation.datastore import DaraStore
from simulation.datastore import DataStoreConfig

from test.audio import AudioChannelConfig
from test.audio import AudioChannel

from dsp.common.qam import QAMConstellation
from dsp.tx.qam import QAMModulator
from dsp.rx.qam import QAMSoftDemodulator

#
# Config
#
ds = DaraStore(config=DataStoreConfig(
    path='out/dumps',
    names=[
        '__main__',
        'test.audio.AudioChannel',
    ]
))

params = SimParams(
    fc=int(4e3),
    fs=int(1e3),
)

RATE = 48000
audio_cfg = AudioChannelConfig(
    fs=RATE,
    channels=1,
    format='Int16',
    chunk=4096,
    tx_zero_prefix=.1,
)

logging.basicConfig(filename='out.log', level=logging.DEBUG)
for module in ['matplotlib', 'PIL']:
    logger = logging.getLogger(module)
    logger.setLevel(level=logging.CRITICAL)


if __name__ == '__main__':

    try:
        #
        # Prepare data
        #
        Nsymb = 100
        Npream = 10
        QAMpow = 2
        payload = np.random.randint(0, 2, Nsymb * QAMpow)

        constellation = QAMConstellation(order=2 ** QAMpow)
        modulator = QAMModulator(constellation=constellation)
        demodulator = QAMSoftDemodulator(constellation=constellation)

        preamble = np.ones(Npream * QAMpow, dtype=np.int32)
        symbols = modulator.process(np.concatenate((preamble, payload)))

        ds.store(tx_symbols=symbols)

        #
        # Move to carrier
        #
        symbols_c = np.repeat(symbols, RATE // params.fs)
        time = np.arange(symbols_c.shape[0]) / RATE
        carrier = np.exp(1j*2*np.pi*params.fc*time)
        signal = np.real(symbols_c * carrier)

        ds.store(tx_signal=signal)

        #
        # Transmit & Receive over the audio devices
        #
        signal_q = (signal * 2**14).astype(np.int16)
        with AudioChannel(config=audio_cfg) as channel:
            recv = channel.process_raw(tx_signal=signal_q)
        ds.store(rx_signal=recv)

        #
        # Process received data
        #

        # Locate Preamble
        preamble_ref = np.repeat(preamble, RATE // params.fs)
        preamble_ref = preamble_ref * carrier[:len(preamble_ref)]

        corr = correlate(recv, preamble_ref)
        corr_offset = np.argmax(np.abs(corr))
        offset = corr_offset - len(preamble_ref)
        ds.store(
            preamble_corr=corr,
            preamble_offset=offset,
        )

        recv = recv[offset:len(time)]
        ds.store(cropped_recv=recv)

        recv_z = recv * np.conj(carrier)
        ds.store(rx_signal_zero_freq=recv_z)

        # matched filtering
        coeffs = np.ones(RATE // params.fs)
        recv_F = lfilter(b=coeffs, a=1, x=recv_z)
        ds.store(rx_signal_filtered=recv_F)

        # Downsample signal
        idxs = np.arange(1, Nsymb + 1) * RATE // params.fs + offset
        symbols_d = recv_F[idxs]
        ds.store(downsampled_symbols=symbols_d)

        # Channel estimation
        chest = symbols[0] * Npream / corr[offset]
        ds.store(channel_est=chest)

        # Channel equalization
        symbols_hat = symbols_d * chest

        payload_hat = demodulator.process(symbols_hat)

        ber = np.sum(payload != payload_hat) / payload.shape[0]
        logging.info(f'Simulation has finished. Estimated BER: {ber}')

    finally:
        ds.flush()
