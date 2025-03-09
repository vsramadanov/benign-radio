import numpy as np
import logging

from scipy.signal import lfilter
from numpy import concatenate as cat

from test.audio import AudioChannelConfig
from test.audio import RawAudioChannel

from simulation.params import SimParams
from simulation.datastore import DataStore

from dsp.common.qam import QAMConstellation
from dsp.common.resampling.fft import FftUpsampler

from dsp.tx.qam import QAMModulator
from dsp.rx.qam import QAMSoftDemodulator

from dsp.tx.ofdm import OFDM, OFDMconfig

from dsp.rx.ofdm.frontend import OFDMfrontend
from dsp.rx.ofdm.equalizer import OFDMChannleEqualizer
from dsp.rx.ofdm.estimator import OFDMChannleEstimator
from dsp.rx.ofdm.chain import OFDMRxChain

# FIR 16 taps, Low pass filter, cutoff = 2 kHz
b_lo = np.array([0.03157319, 0.04156552, 0.05169825, 0.06138433, 0.07002734,
                 0.07707174, 0.08205098, 0.08462865, 0.08462865, 0.08205098,
                 0.07707174, 0.07002734, 0.06138433, 0.05169825, 0.04156552,
                 0.03157319])

# FIR 16 taps, Band pass filter, F_lo = 3 kHz F_hi = 5 kHz
b_bp = np.array([-0.06280036, -0.09516154, -0.10379872, -0.0815819, -0.03158135,
                 0.03292086,  0.09244693,  0.12802351,  0.12802351,  0.09244693,
                 0.03292086, -0.03158135, -0.0815819, -0.10379872, -0.09516154,
                 -0.06280036])


class Scenario:

    def run(self, config: dict):

        params = SimParams()
        ds = DataStore()

        scenario_cfg = config['scenario']
        audio_cfg = AudioChannelConfig(**config['AudioChannelConfig'])

        Nsymb = scenario_cfg['Nsymb']
        payload_len = Nsymb * config['OFDMconfig']['Nsc'] * \
            int(np.log2(scenario_cfg['constellation']['order']))
        audio_upscale_factor = audio_cfg.rate // params.fs
        logging.info(f'\t Symbols amount: {Nsymb}')
        logging.info(f'\t bit len: {payload_len}')
        logging.info(f'\t Audio upscale factor: {audio_upscale_factor}')

        logging.info(f'============ Build up a transmitter ============')
        ofdm_config = OFDMconfig(**config['OFDMconfig'])
        ofdm_modulator = OFDM(config=ofdm_config)

        constellation = QAMConstellation(**scenario_cfg['constellation'])
        modulator = QAMModulator(constellation=constellation)

        upsampler = FftUpsampler(scale=audio_upscale_factor)

        payload = np.random.randint(0, 2, payload_len)
        tx_symbols = modulator.process(payload)
        ofdm_signal = ofdm_modulator.process(tx_symbols)
        ds.store(payload=payload, tx_symbols=tx_symbols,
                 ofdm_signal=ofdm_signal)

        # Upconversion
        envelope = upsampler.process(ofdm_signal)
        time = np.arange(len(envelope)) / audio_cfg.rate
        ref = np.exp(-1j*2*np.pi*params.fc*time)
        iq_signal = envelope * ref
        signal = np.real(iq_signal)
        ds.store(tx_envelope=envelope)

        #
        # TODO: Transmit & Receive over the audio device
        #
        prefix_len = 0  # np.random.randint(100, 1000)
        suffix_len = 0  # np.random.randint(100, 1000)
        recv_iq_signal = np.pad(signal, (prefix_len, suffix_len))
        ds.store(tx_signal=signal,
                 prefix_len=prefix_len,
                 suffix_len=suffix_len,
                 recv_iq_signal=recv_iq_signal)

        #
        # Receiver
        #
        logging.info(f'============ Build up a receiver ============')
        ofdm_rx_chain = OFDMRxChain(
            frontend=OFDMfrontend(config=ofdm_config),
            estimator=OFDMChannleEstimator(),
            equalizer=OFDMChannleEqualizer(),
        )
        demodulator = QAMSoftDemodulator(constellation=constellation)

        # Filtering
        filt_iq_signal = lfilter(
            x=cat((recv_iq_signal, np.zeros(len(b_bp)))),
            b=b_bp,
            a=1)

        # Downconversion
        time = np.arange(len(filt_iq_signal)) / audio_cfg.rate
        ref = np.exp(-1j*2*np.pi*params.fc*time)
        downconv_iq_signal = filt_iq_signal * ref
        filtered_iq_signal = lfilter(
            x=cat((downconv_iq_signal, np.zeros(len(b_lo)))),
            b=b_lo,
            a=1)
        rx_ofdm_signal = filtered_iq_signal[15::audio_upscale_factor]
        ds.store(downconv_iq_signal=downconv_iq_signal,
                 rx_ofdm_signal=rx_ofdm_signal)

        # OFDM demodulation
        rx_symbols = ofdm_rx_chain.process(rx_ofdm_signal)
        ds.store(rx_symbols=rx_symbols)

        # symbol demodulation
        payload_hat = demodulator.process(rx_symbols)

        ber = np.sum(
            np.abs(payload - payload_hat)) / payload.shape[0]
        logging.info(f'Simulation has finished. Estimated BER: {ber}')
