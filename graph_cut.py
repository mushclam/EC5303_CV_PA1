import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from pygco import cut_simple, cut_from_graph

def graph_cut(input_image, scribble, matrix, n_label, n_modal):
    width, height = input_image.shape
    # input image pre-processing
    input_image = input_image[np.newaxis, :, :]
    # scribble pre-processing
    scribbles = np.array([
        np.where(scribble[:, :, 2] == i+1, input_image, 0.) for i in range(n_label)
    ]).reshape(n_label, -1)

    # multi-modality
    if n_modal == 1:
        mean = np.mean(scribbles, where=(
            scribbles != 0), axis=1, keepdims=True)
        var = np.var(scribbles, where=(
            scribbles != 0), axis=1, keepdims=True)
        mean = mean[:, :, np.newaxis]
        var = var[:, :, np.newaxis]

        total_cost = (((input_image-mean)**2) / (2*var))
    elif n_modal > 1:
        n_modals = n_modal * np.ones(n_label, dtype=np.int32)
        total_cost = []
        for i, nm in enumerate(n_modals):
            s = scribbles[i]
            s = s[np.nonzero(s)]
            if nm == 1:
                mean = np.mean(s, keepdims=True)
                var = np.var(s, keepdims=True)
            elif nm > 1:
                mean = []
                var = []
                s_min = s.min()
                gap = (np.ptp(s)+1) / nm
                for n in range(nm):
                    part_s = s[np.logical_and(
                        s >= (s_min+n*gap), s < (s_min+(n+1)*gap))]
                    if part_s.size != 0:
                        mean.append(np.mean(part_s))
                        var.append(np.var(part_s))
                    else:
                        mean.append(0)
                        var.append(0)
                mean = np.array(mean)
                var = np.array(var)
            mean = mean[:, np.newaxis, np.newaxis]
            var = var[:, np.newaxis, np.newaxis]
            c = (-(input_image-mean)**2) / (2*var)
            c = -np.log(np.sum(np.exp(c), axis=0))
            total_cost.append(c)
        total_cost = np.array(total_cost)

    # edges = np.array(
    #     [matrix.row, matrix.col, matrix.data]
    # ).transpose(1, 0).copy("C").astype(np.int32)

    # test for neigborhood matrix
    total_cost = total_cost.transpose(1, 2, 0)
    total_cost = total_cost.reshape(-1, n_label)
    identity_matrix = sparse.eye(matrix.shape[0])
    matrix = identity_matrix + matrix
    total_cost = matrix @ total_cost
    total_cost = total_cost.reshape(width, height, n_label).copy("C").astype(np.int32)

    # graph-cut
    if n_label == 2:
        pairwise = -10 * np.eye(n_label, dtype=np.int32)
    elif n_label == 7:
        x, y = np.ogrid[:n_label, :n_label]
        pairwise = 5 * np.abs(x - y).astype(np.int32).copy("C")

    result = cut_simple(total_cost, pairwise)
    return result[np.newaxis, :, :]
