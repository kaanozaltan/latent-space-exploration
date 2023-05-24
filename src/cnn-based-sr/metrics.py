import math

import numpy as np


def psnr(outputs, targets):
    mse = np.mean((outputs - targets) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
