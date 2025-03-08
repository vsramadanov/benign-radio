import numpy as np
import logging


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

        # Downconversion
        downconv_iq_signal = recv_iq_signal * np.conj(ref)
        rx_ofdm_signal = downconv_iq_signal[::audio_upscale_factor]
        ds.store(rx_ofdm_signal=rx_ofdm_signal)
        
        # OFDM demodulation
        rx_symbols = ofdm_rx_chain.process(rx_ofdm_signal)
        ds.store(rx_symbols=rx_symbols)

        # symbol demodulation
        payload_hat = demodulator.process(rx_symbols)

        ber = np.sum(
            np.abs(payload - payload_hat)) / payload.shape[0]
        logging.info(f'Simulation has finished. Estimated BER: {ber}')
