import pytest
import phiplot as pp
import pyphi
import numpy as np

@pytest.fixture
def ocx_net():
    tpm = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]])

    cm = np.array([[0, 0, 1],
                   [1, 0, 1],
                   [1, 1, 0]])

    node_labels = ('P', 'Q', 'R')
    return pyphi.Network(tpm, cm, node_labels)
