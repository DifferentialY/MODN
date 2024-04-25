import torch
import numpy as np


def RouletteWheelSelection(P):
    r = torch.rand(1)
    C = torch.cumsum(P, dim=1)
    Cnum = np.size(np.array(P), 1)
    for i in range(Cnum):
        if r <= C[0, i]:
            j = i
            break

    return j