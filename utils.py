from IPython import get_ipython
import matplotlib.pyplot as plt

import numpy as np

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def plot_examples(n, data, cmap):
    if n == 1:
        fig = plt.figure(figsize=(5, 3))
        psm = plt.pcolormesh(data, cmap=cmap,
                             rasterized=True, vmin=0, vmax=1)
        plt.colorbar(psm)
        plt.gca().invert_yaxis()
        plt.show()
        plt.clf()
    elif n > 1:
        fig, axs = plt.subplots(1, n, figsize=(n*5, 3),
                                constrained_layout=True, squeeze=False)
        for [ax, s] in zip(axs.flat, data):
            psm = ax.pcolormesh(
                s, cmap=cmap, rasterized=True)
            ax.invert_yaxis()
            plt.colorbar(psm, ax=ax)
        plt.show()
        plt.clf()
    else:
        raise ValueError("We can't not print negative plots.")