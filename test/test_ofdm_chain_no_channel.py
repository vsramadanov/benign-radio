import pytest
import logging
import numpy as np

from simulation.params import SimParams
from simulation.datastore import DaraStore
from simulation.datastore import DataStoreConfig

from dsp.tx.ofdm import OFDM, OFDMconfig, GItype

from dsp.rx.ofdm.frontend import OFDMfrontend
from dsp.rx.ofdm.equalizer import OFDMChannleEqualizer
from dsp.rx.ofdm.estimator import OFDMChannleEstimator
from dsp.rx.ofdm.chain import OFDMRxChain

Nsymb = 1000  # OFDM symbols to simulate
Scs = 1e3  # subcarrier spacing


@pytest.mark.quick
@pytest.mark.parametrize("fs", (6e3, 12e3,))
def test_ofdm_chain(fs):
    logging.basicConfig(level=logging.DEBUG)

    SimParams(
        fc=10e3,
        fs=fs,
    )

    DaraStore(config=DataStoreConfig(
        path='out/dumps',
        names=[]
    ))

    Nsc = int(SimParams().fs / Scs)  # number of subcarriers
    test_ofdm_config = OFDMconfig(
        Ncs=Nsc,
        GI=4,
        Type=GItype.CP,
    )

    ofdm_modulator = OFDM(config=test_ofdm_config)
    ofdm_rx_chain = OFDMRxChain(
        frontend=OFDMfrontend(config=test_ofdm_config),
        estimator=OFDMChannleEstimator(),
        equalizer=OFDMChannleEqualizer(),
    )

    bitstream = np.random.randint(0, 2, Nsc * Nsymb)

    tx_symbols = 2*bitstream - 1

    ofdm_signal = ofdm_modulator.process(tx_symbols)

    recv_ofdm_signal = ofdm_signal  # propagation

    rx_symbols = ofdm_rx_chain.process(recv_ofdm_signal)

    # symbol demodulation
    bitstream_hat = (np.real(rx_symbols) > 0).astype(np.int64)

    assert np.all(bitstream == bitstream_hat)  # no noise no error
