from tqdm import tqdm
import numpy as np
import pickle

from scipy import sparse
from scipy.sparse import linalg

import cupy as cp
from cupyx import scipy as cs
from cupyx.scipy.sparse import linalg as culinalg

import torch

# pyTorch do not have sparse matrix operation
def _torch_least_square_solution(scribble, matrix, device=torch.device('cuda:0')):
    size = matrix.shape[0]
    scribble = torch.tensor(scribble)
    scribble = scribble.reshape(-1, 3)
    scribbles = [
        torch.where(scribble[:, 2] == 1, 1., 0.).double(),
        torch.where(scribble[:, 2] == 2, 1., 0.).double()
    ]

    result = []
    batch_size = 1024
    for i, s in enumerate(scribbles):
        s = s.to(device)
        indices = []
        d = []
        for j in tqdm(range(size//batch_size), desc=f'Scribble {i}'):
            dense_matrix = split_coo_matrix_to_dense(
                matrix, batch_size*j, batch_size*(j+1)
            )
            x = torch.linalg.lstsq(
                dense_matrix.to(device),
                s).solution
            if torch.any(x):
                indices.append(
                    (torch.where(x != 0)[0] + batch_size*j).to('cpu'))
                d.append(x.to('cpu'))

        remain = size % batch_size
        dense_matrix = split_coo_matrix_to_dense(
            matrix, batch_size*j, batch_size*j+remain
        )
        x = torch.linalg.lstsq(
            dense_matrix.to(device),
            s).solution
        if torch.any(x):
            indices.append((torch.where(x != 0)[0] + batch_size*j).to('cpu'))
            d.append(x.to('cpu'))

        indices = torch.cat(indices)
        d = torch.cat(d)

        # total = torch.sparse_coo_tensor(
        #     indices=[torch.ones(len(indices)), indices],
        #     values=d,
        #     size=[1, size]
        # )
        result.append((indices, d))
    with open('tmp_pred_scribble_label.pkl', 'wb') as f:
        pickle.dump(result, f)
    result = torch.cat(result)
    return result


# Perform without error
def least_square_solution(operator, scribble, matrix, precision):
    # Original Shape
    width, height = scribble.shape[:2]
    # Matrix pre-processing
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = sparse.csr_matrix(matrix)
    identity_matrix = sparse.eye(matrix.shape[0])
    matrix = identity_matrix - matrix
    
    # Scribble pre-processing
    scribble = scribble.reshape(-1, 3)
    scribbles = [
        np.where(scribble[:, 2] == i+1, 1., 0.) for i in range(scribble.max())
    ]
    scribbles = sparse.csr_matrix(scribbles)

    # Least-square solution
    result = []
    for s in tqdm(scribbles, desc='Least Square Solution'):
        print('\n[Update neighborhood matrix to fit with scribble]')
        s_matrix = matrix.copy()
        for ind in s.indices:
            tmp_mat = s_matrix.getrow(ind).ceil()
            tmp_mat.eliminate_zeros()
            s_matrix[ind] = tmp_mat
        print('[Calculate Least-Square Solution]')
        s = s.transpose().toarray().squeeze()
        if operator == 'lsqr':
            x, *_ = linalg.lsqr(
                s_matrix, s,
                show=True,
                atol=precision,
                btol=precision,
                conlim=0
            )
        elif operator == 'lsmr':
            x, *_ = linalg.lsmr(
                matrix, s,
                show=True,
                atol=precision,
                btol=precision,
                conlim=0
            )
        else:
            raise TypeError()
        result.append(x[np.newaxis, :])
    result = np.concatenate(result, axis=0).reshape(scribbles.shape[0], width, height)
    return np.expand_dims(np.argmax(result, axis=0), axis=0)


# Out of Memory
def _cupyx_least_square_solution(scribble, matrix):
    if not isinstance(matrix, cs.sparse.coo_matrix):
        matrix = cs.sparse.coo_matrix(matrix)
    scribble = scribble.reshape(-1, 3)
    scribbles = cp.array([
        np.where(scribble[:, 2] == 1, 1., 0.),
        np.where(scribble[:, 2] == 2, 1., 0.)
    ])
    result = []
    for s in tqdm(scribbles, desc='Least Square Solution'):
        x, *_ = culinalg.lsqr(matrix, s)
        result.append(x)
    return np.concatenate(result)


def split_coo_matrix_to_dense(m, p, q):
    mat = []
    for i in range(p, q):
        mat.append(m[i].to_dense().unsqueeze(1))
    return torch.cat(mat, dim=1)
