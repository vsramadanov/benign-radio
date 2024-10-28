import numpy as np
import logging

from scipy.signal import lfilter

from test.audio import AudioChannelConfig
from test.audio import RawAudioChannel

from simulation.params import SimParams
from simulation.datastore import DataStore

from dsp.common.qam import QAMConstellation
from dsp.tx.qam import QAMModulator
from dsp.rx.qam import QAMSoftDemodulator


class Scenario:

    def run(self, config: dict):

        params = SimParams()
        ds = DataStore()

        audio_cfg = AudioChannelConfig(**config['AudioChannelConfig'])

        #
        # Prepare data
        #
        Nsymb = 1000
        Npreamb = 25
        QAMpow = 2
        SFlen = audio_cfg.rate // params.fs  # shaping filter len
        payload = np.random.randint(0, 2, Nsymb * QAMpow)

        constellation = QAMConstellation(order=2 ** QAMpow)
        modulator = QAMModulator(constellation=constellation)
        demodulator = QAMSoftDemodulator(constellation=constellation)

        preamble_bits = np.random.randint(0, 2, Npreamb * QAMpow)
        symbols = modulator.process(np.concatenate((preamble_bits, payload)))
        preamble = symbols[:Npreamb]

        ds.store(tx_symbols=symbols)

        #
        # Move to carrier
        #
        symbols_rep = np.repeat(symbols, SFlen)
        time = np.arange(len(symbols_rep)) / audio_cfg.rate
        carrier = np.exp(-1j*2*np.pi*params.fc*time)
        iq_signal = symbols_rep * carrier
        signal = np.real(iq_signal)

        logging.info(f'transmittion time: {time[-1]}')
        ds.store(carrier=carrier, tx_signal=iq_signal)

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

        # matched filtering
        time = np.arange(SFlen) / audio_cfg.rate
        coeffs = np.conj(np.exp(-1j*2*np.pi*params.fc*time[-1::-1]))

        recv_F = lfilter(b=coeffs, a=1, x=recv)
        ds.store(mf_coeffs=coeffs, rx_signal_filtered=recv_F)

        # Locate Preamble
        # suppose preamble located in the first 20% bins length SFlen
        front_idx = (len(recv_F) // SFlen) // 5 * SFlen
        front_recv = recv_F[:front_idx].reshape((-1, SFlen))
        corr = lfilter(b=np.conj(preamble[-1::-1]), a=1, x=front_recv, axis=0)
        corr_abs = np.abs(corr)

        row_idx = np.argmax(corr_abs, axis=0)
        col_idx = np.argmax(corr_abs[row_idx, np.arange(SFlen)])
        offset = row_idx[col_idx] * SFlen + col_idx
        max_corr = corr[row_idx[col_idx], col_idx]
        logging.info(
            f"located preamble at {offset - SFlen * Npreamb} offset, correlation: {max_corr}")
        ds.store(
            front_rx_signal=front_recv,
            preamble_corr=corr,
            preamble_offset=offset,
        )

        symb_idx = offset + SFlen + np.arange(Nsymb) * SFlen
        symbols_d = recv_F[symb_idx]
        ds.store(downsampled_symbols=symbols_d)

        # Channel estimation
        chest = 2 * Npreamb / max_corr
        logging.info(
            f'estimated channel H: {chest}, angle: {np.rad2deg(np.angle(chest))}')

        # Channel equalization
        symbols_hat = symbols_d * chest
        ds.store(symbols_hat=symbols_hat)

        payload_hat = demodulator.process(symbols_hat)

        ber = np.sum(
            np.abs(payload - payload_hat)) / payload.shape[0]
        logging.info(f'Simulation has finished. Estimated BER: {ber}')
