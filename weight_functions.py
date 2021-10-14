import numpy as np


def weight_func1(r, s, var):
    w = np.exp(-(r.astype(np.int32)-s.astype(np.int32))**2 / (2*var))
    return w / np.sum(w)


def weight_func2(r, s, mean, var):
    w = 1 + (r.astype(np.int32) - mean)*(s.astype(np.int32) - mean)/var
    return w / np.sum(w)

def laplacian(r_coord, r, s_coord, s):
    x_r, y_r = r_coord

    g = r.astype(np.int32) - s.astype(np.int32)
    return g