import pytest
import logging
import numpy as np

from simulation.params import SimParams
from simulation.datastore import DataStore
from simulation.datastore import DataStoreConfig

from dsp.common.qam import QAMConstellation
from dsp.tx.qam import QAMModulator
from dsp.rx.qam import QAMSoftDemodulator

Nsymb = 1000  # QAM symbols to simulate


@pytest.mark.quick
@pytest.mark.parametrize("order", (2, 4, 16))
def test_qam_modulation_demodulation(order):
    logging.basicConfig(level=logging.DEBUG)

    SimParams(
        fc=10e3,  # no need, but has to configure
        fs=100e3,
    )

    DataStore(config=DataStoreConfig(
        path='out/dumps',
        names=[]
    ))

    constellation = QAMConstellation(order=order)
    modulator = QAMModulator(constellation=constellation)
    demodulator = QAMSoftDemodulator(constellation=constellation)

    bitstream = np.random.randint(0, 2, Nsymb * constellation.pow)
    symbols = modulator.process(bitstream)
    bitstream_hat = demodulator.process(symbols)

    assert np.all(bitstream == bitstream_hat)  # no noise no error
