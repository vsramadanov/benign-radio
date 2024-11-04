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
        ofdm_rate = scenario_cfg['ofdm_rate']  # OFDM symbols per sec
        points_per_ofdm_symbol = config['OFDMconfig']['Nsc'] + \
            config['OFDMconfig']['guard_interval_length']
        k = audio_cfg.rate / ofdm_rate / points_per_ofdm_symbol

        ofdm_config = OFDMconfig(**config['OFDMconfig'])
        ofdm_modulator = OFDM(config=ofdm_config)

        constellation = QAMConstellation(order=4)
        modulator = QAMModulator(constellation=constellation)

        payload = np.random.randint(0, 2, ofdm_config.Nsc * Nsymb)
        tx_symbols = modulator.process(payload)
        ofdm_signal = ofdm_modulator.process(tx_symbols)
        ds.store(payload=payload, tx_symbols=tx_symbols,
                 ofdm_signal=ofdm_signal)

        # Upconversion
        symbols_rep = np.repeat(ofdm_signal, k)
        time = np.arange(len(symbols_rep)) / audio_cfg.rate
        carrier = np.exp(-1j*2*np.pi*params.fc*time)
        iq_signal = symbols_rep * carrier
        signal = np.real(iq_signal)

        #
        # Transmit & Receive over the audio devices
        #
        signal_q = (signal * 2**14).astype(np.int16)
        with RawAudioChannel(config=audio_cfg) as channel:
            recv_ofdm_signal = channel.route_audio(input=signal_q)
        ds.store(rx_signal=recv_ofdm_signal)
