#%%
from skimage import io
import cv2

import numpy as np

from matplotlib import cm

import os
import pickle
import argparse

from neighborhood import *
from utils import *
from least_square_solution import *
from graph_cut import *
from metric import *


def init_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', '-d', default='data/')
    parser.add_argument('--matrix_dir', '-md', default='matrix/')
    parser.add_argument('--scribble_dir', '-sd', default='scribble/')
    parser.add_argument('--gpu', '-g', type=int, default=0)

    parser.add_argument('--n_label', '-nl', type=int, default=2, choices=[2, 7])
    parser.add_argument('--method', '-m', type=str, default='graph-cut', choices=['lstsq', 'graph-cut'])
    parser.add_argument('--operator', '-o', type=str, default='none', choices=['none', 'lsqr', 'lsmr'])
    parser.add_argument('--threshold', '-t', type=float, default=8)
    parser.add_argument('--precision', '-p', type=float, default=0)
    parser.add_argument('--n_modal', '-nm', type=int, default=1)
    parser.add_argument('--weight-function', '-wf', type=str, default='laplacian', choices=['w1', 'w2', 'laplacian'])

    if is_notebook():
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


def main():
    args = init_argument()

    if not os.path.exists(args.matrix_dir):
        os.mkdir(args.matrix_dir)

    if not os.path.exists(args.scribble_dir):
        os.mkdir(args.scribble_dir)

    method = args.method
    operator = args.operator
    precision = args.precision
    threshold = args.threshold
    weight_function = args.weight_function
    n_label = args.n_label
    n_modal = args.n_modal
    
    # Check existence of scribble file
    if method == 'lstsq':
        scribble_name = f'{operator}_t{threshold}_{weight_function}_p{precision:.0e}_l{n_label}_scribble.pkl'
    elif method == 'graph-cut':
        scribble_name = f'{method}_t{threshold}_{weight_function}_l{n_label}_m{n_modal}_scribble.pkl'
    scribble_name = os.path.join(args.scribble_dir, scribble_name)

    if os.path.exists(scribble_name):
        # If exist, load scribble
        with open(scribble_name, 'rb') as f:
            predicted = pickle.load(f)
    else:
        # If not exist, check existence of neighborhood matrix
        file_name = f'{threshold}_{weight_function}_matrix.pkl'
        file_name = os.path.join(args.matrix_dir, file_name)

        if os.path.exists(file_name):
            # If exist, load matrix
            with open(file_name, 'rb') as f:
                matrix = pickle.load(f)
        else:
            # If not exist, generate and save neighborhood matrix
            image = io.imread('data/Emily-In-Paris-gray.png')
            matrix = neighborhood_weight(image, threshold, weight_function)
            with open(file_name, 'wb') as f:
                pickle.dump(matrix, f)

        if n_label == 2:
            scribble_image = cv2.imread(
                'data/Emily-In-Paris-scribbles.png')
        elif n_label == 7:
            scribble_image = cv2.imread(
                'data/Emily-In-Paris-scribbles-plus.png')
        
        # After load(or generate) neighborhood matrix
        # Generate and save scribble
        if method == 'lstsq':
            # Least-Square Solution
            predicted = least_square_solution(
                args.operator, scribble_image, matrix,
                precision, args
            )
        elif method == 'graph-cut':
            # load input image
            input_image = io.imread('data/Emily-In-Paris-gray.png')
            # graph-cut
            predicted = graph_cut(input_image, scribble_image, matrix, n_label, n_modal)
        # Save predicted scribble
        with open(scribble_name, 'wb') as f:
            pickle.dump(predicted, f)

    # Colormap
    if n_label == 2:
        gt = cv2.imread('data/Emily-In-Paris-gt.png')[:, :, 2]
        cmap = cm.get_cmap('gray', n_label)
    elif n_label == 7:
        gt = cv2.imread('data/Emily-In-Paris-gt-plus.png')[:, :, 2]
        cmap = cm.get_cmap('viridis', n_label)
        # For matching labels
        predicted = predicted + 1
    # Print
    print(f'[{scribble_name}]')
    plot_examples(
        2,
        np.concatenate([
            predicted,
            gt[np.newaxis, :, :]
        ], axis=0),
        cmap=cmap,
    )
    # Calculate IoU
    calculate_IoU(n_label, predicted, gt)

main()

# %%
