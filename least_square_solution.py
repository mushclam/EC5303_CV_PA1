from tqdm import tqdm
import numpy as np

from scipy import sparse
from scipy.sparse import linalg

import os
import pickle

def least_square_solution(operator, scribble, matrix, precision, wf):
    # Original Shape
    width, height = scribble.shape[:2]
    # Matrix pre-processing
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = sparse.csr_matrix(matrix)
    if wf != 'laplacian':
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
    for i, s in tqdm(enumerate(scribbles), desc='Least Square Solution'):
        print('\n[Update neighborhood matrix to fit with scribble]')
        # Generate and save reconstructed matrix that need many time
        tmp_f_name = f'tmp/tmp_{operator}_t16_lap_l2_{i}_mat.pkl'
        if os.path.exists(tmp_f_name):
            with open(tmp_f_name, 'rb') as f:
                s_matrix = pickle.load(f)
            print('[Load Complete]')
        else:
            s_matrix = matrix.copy()
            for ind in s.indices:
                tmp_mat = s_matrix.getrow(ind).ceil()
                tmp_mat.eliminate_zeros()
                s_matrix[ind] = tmp_mat
            with open(tmp_f_name, 'wb') as f:
                pickle.dump(s_matrix, f)
        # LSTSR by using reconstructed matrix
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