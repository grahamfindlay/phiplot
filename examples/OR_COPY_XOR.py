import numpy as np
import pyphi

# Big Phi = 2.3125

tpm = np.array([[0,0,0],
                [0,0,1],
                [1,0,1],
                [1,0,0],
                [1,1,0],
                [1,1,1],
                [1,1,1],
                [1,1,0]])

cm = np.array([[0,0,1],
               [1,0,1],
               [1,1,0]])

node_labels = ('P', 'Q', 'R')

state = (1, 0, 0)

def net():
    return pyphi.Network(tpm, cm, node_labels)

def sub():
    return pyphi.Subsystem(net(), state, node_labels)
