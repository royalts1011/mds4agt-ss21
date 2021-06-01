import numpy
import numpy as np


def compute_points(angles, step_sz=0.7):
    x = np.zeros_like(angles)
    y = np.zeros_like(angles)

    for i in range(1,len(x)):
        x[i] = x[i-1] + step_sz * np.cos(angles[i-1])
        y[i] = y[i-1] + step_sz * np.sin(angles[i-1])

    return x, y

