import pytest

import numpy as np
import matplotlib.pyplot as plt

from dsp.common.qam import QAMConstellation


@pytest.mark.visual
@pytest.mark.parametrize("order", (2, 4, 16, ))
def test_qam_eye_view(order):
    qam = QAMConstellation(order=order)

    constellation = qam.mapping

    for symbol, val in constellation:
        plt.plot(np.real(symbol), np.imag(symbol),
                 marker='o', color='k', linestyle='none')
        plt.annotate(f"{val}", (np.real(symbol), np.imag(symbol)))

    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(f"QAM{order} constellation")
    plt.grid(True)
    plt.show()
