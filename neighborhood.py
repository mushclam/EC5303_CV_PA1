import numpy as np
from scipy import sparse
from tqdm import tqdm

from weight_functions import *

def neighborhood_weight(image, threshold, wf):
    row_length = image.shape[0]
    col_length = image.shape[1]
    size = row_length * col_length
    row_indices = []
    col_indices = []
    data = []

    # Calculate weights of neighborhoods
    n_indices = get_neighbor_indices(threshold)
    for i, row in tqdm(enumerate(image), desc='Neighbor Matrix'):
        for j, r in enumerate(row):
            # i, j: coordinate of center pixel
            # r: intensity of center pixel
            # n: a list of neighborhood pixels
            # s: a list of intensities of neighborhood pixels
            center_index = i*col_length + j

            n = neighborhood_pixel(i, j, row_length, col_length, n_indices)
            s = np.array([image[x, y] for (x, y) in n])
            neighbor_indices = np.array([x*col_length+y for (x, y) in n])

            if wf == 'w1' or wf == 'w2':
                mean = np.mean(s)
                var = np.var(s)
                if var == 0:
                    w = np.ones_like(s) / len(s)
                else:
                    if wf == 'w1':
                        w = weight_func1(r, s, var)
                    else:
                        w = weight_func2(r, s, mean, var)
            elif wf == 'laplacian':
                w = laplacian((i, j), r, n, s)

            for index, e in enumerate(neighbor_indices):
                row_indices.append(center_index)
                col_indices.append(e)
                data.append(w[index])

    # Generate neighborhood matrix
    return sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(size, size)
    )


def get_neighbor_indices(threshold):
    indices = set([])
    i, j = 0, 0
    while True:
        while True:
            if i == 0 and j == 0:
                j += 1
                continue
            flag = np.linalg.norm([i, j])
            if flag <= threshold:
                indices.add((i, j))
                j += 1
            else:
                break
        if j == 0:
            break
        else:
            i += 1
            j = 0
    i_neg = [(-i, j) for i, j in indices]
    j_neg = [(i, -j) for i, j in indices]
    both_neg = [(-i, -j) for i, j in indices]
    indices.update(i_neg)
    indices.update(j_neg)
    indices.update(both_neg)

    return list(indices)


def neighborhood_pixel(x, y, width, height, n_indices):
    abs_indices = []
    for index in n_indices:
        i, j = x + index[0], y + index[1]
        if i < 0 or j < 0:
            continue
        elif i >= width or j >= height:
            continue
        else:
            abs_indices.append((i, j))

    return abs_indices
