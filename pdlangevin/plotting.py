"""Helpers for saving image results as borderless PDFs, as used in the paper."""
import os

import matplotlib.pyplot as plt
import numpy as np


def save_image(im, path, vmin=0, vmax=1, size=4):
    """Save a grayscale image as a borderless square figure."""
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.asarray(im), cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, bbox_inches=0)
    plt.close(fig)


def save_colorbar(path, vmin, vmax):
    """Save a standalone vertical gray colorbar for the range [vmin, vmax]."""
    fig = plt.figure(figsize=(0.5 * 0.6 * 6.15, 0.5 * 3.1 * 6.15))
    ax = fig.add_axes((0.05, 0.05, 0.2, 0.9))
    sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, cax=ax)
    ax.tick_params(axis='y', labelsize=30)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def result_dir(name, root=None):
    """Return (and create) the directory ``<root>/<name>``, default root ``results/``."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results') if root is None else root
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def data_path(filename):
    """Absolute path of a file in the repository's ``data/`` directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', filename)
