import numpy as np
import logging

from scipy.signal import lfilter
from scipy.signal import correlate

import matplotlib.pyplot as plt

from simulation.params import SimParams
from simulation.datastore import DaraStore
from simulation.datastore import DataStoreConfig

from test.audio import AudioChannelConfig
from test.audio import RawAudioChannel

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
    rate=RATE,
    channels=1,
    format='Int16',
    device="default"
)

logging.basicConfig(filename='out/audio_sim.log', level=logging.DEBUG)
for module in ['matplotlib', 'PIL']:
    logger = logging.getLogger(module)
    logger.setLevel(level=logging.CRITICAL)


if __name__ == '__main__':

    try:
        #
        # Prepare data
        #
        Nsymb = 1000
        Npream = 10
        QAMpow = 2
        SFlen = RATE // params.fs  # shaping filter len
        payload = np.random.randint(0, 2, Nsymb * QAMpow)

        constellation = QAMConstellation(order=2 ** QAMpow)
        modulator = QAMModulator(constellation=constellation)
        demodulator = QAMSoftDemodulator(constellation=constellation)

        preamble = np.random.randint(0, 2, Npream * QAMpow)
        symbols = modulator.process(np.concatenate((preamble, payload)))

        ds.store(tx_symbols=symbols)

        #
        # Move to carrier
        #
        symbols_c = np.repeat(symbols, SFlen)
        time = np.arange(symbols_c.shape[0]) / RATE
        carrier = np.exp(1j*2*np.pi*params.fc*time)
        iq_signal = symbols_c * carrier
        signal = np.real(iq_signal)

        ds.store(tx_signal=signal)

        #
        # Transmit & Receive over the audio devices
        #

        signal_q = (signal * 2**14).astype(np.int16)
        with RawAudioChannel(config=audio_cfg) as channel:
            recv = channel.route_audio(input=signal_q)

        ds.store(rx_signal=recv)

        #
        # Process received data
        #

        # Locate Preamble
        preamble_len = Npream * SFlen
        preamble_ref = iq_signal[:preamble_len]

        corr = correlate(recv, preamble_ref)
        corr_idx = np.argmax(np.abs(corr))
        offset = corr_idx - preamble_len
        logging.info(
            f"located preamble at {offset} offset, correlation: {corr[corr_idx]}")
        ds.store(
            preamble_corr=corr,
            preamble_offset=offset,
        )

        lo = offset
        hi = offset + symbols_c.shape[0] + SFlen
        recv = recv[lo:hi]
        ds.store(cropped_recv=recv)

        time = np.arange(recv.shape[0]) / RATE
        carrier = np.exp(1j*2*np.pi*params.fc*time)
        recv_z = recv * np.conj(carrier)
        ds.store(rx_signal_zero_freq=recv_z)

        # matched filtering
        coeffs = np.ones(SFlen)
        recv_F = lfilter(b=coeffs, a=1, x=recv_z)
        ds.store(rx_signal_filtered=recv_F)
        recv_F = recv_F[SFlen:]  # remove shaping filter impulse responce

        # Downsample signal
        idxs = np.arange(Npream + Nsymb) * SFlen
        symbols_d = recv_F[idxs]
        ds.store(downsampled_symbols=symbols_d)

        # Channel estimation
        chest = Npream * SFlen / corr[corr_idx]
        logging.info(f'estimated channel H: {chest}')

        # Channel equalization
        symbols_hat = symbols_d * chest
        ds.store(symbols_hat=symbols_hat)

        payload_hat = demodulator.process(symbols_hat)

        ber = np.sum(
            np.abs(payload - payload_hat[Npream * QAMpow:])) / payload.shape[0]
        logging.info(f'Simulation has finished. Estimated BER: {ber}')

    finally:
        ds.flush()
