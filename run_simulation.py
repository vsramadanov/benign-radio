import numpy as np

from simulation.params import SimParams
from channel.path_loss import PathLossChannel

from dsp.tx.ofdm import OFDM, OFDMconfig, GItype

from dsp.rx.ofdm.frontend import OFDMfrontend
from dsp.rx.ofdm.equalizer import OFDMChannleEqualizer
from dsp.rx.ofdm.estimator import OFDMChannleEstimator
from dsp.rx.ofdm.chain import OFDMRxChain

SimParams(
    fc=10e9,
    fs=25e6,
)

example_ofdm = OFDMconfig(
    Ncs=12,
    GI=4,
    Type=GItype.CP,
)

ofdm_modulator = OFDM(config=example_ofdm)

data = np.random.randint(0, 2, 24)
print("data=", data)

# Transmitter

# symbol modulation
tx_symbols = 2*data - 1
print("bpsk=", tx_symbols)

ofdm_signal = ofdm_modulator.process(tx_symbols)

print("ofdm=", ofdm_signal)
print("ofdm.shape=", ofdm_signal.shape)

recv_ofdm_signal = ofdm_signal

# Receiver
receiver = OFDMRxChain(
    frontend=OFDMfrontend(config=example_ofdm),
    estimator=OFDMChannleEstimator(),
    equalizer=OFDMChannleEqualizer(),
)

rx_symbols = receiver.process(recv_ofdm_signal)

# symbol demodulation
rx_data = ((rx_symbols / np.abs(rx_symbols) + 1) / 2).astype(np.int64)

assert np.all(data == rx_data)  # no noise no error
