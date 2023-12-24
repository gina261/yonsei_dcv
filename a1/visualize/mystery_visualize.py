#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors, src=None, tgt=None):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        X: an HxW matrix
        colors: an Nx3 color map of colors in [0,1] [R,G,B]
        src: path to X
        tgt: save your visualizations here
    Outputs:
        an HxW uint8 image using the given colormap
    """

    return None


if __name__ == "__main__":
    # load data
    src = "mysterydata/mysterydata.npy"
    data = np.load(src)

    # load colors
    colors = np.load("mysterydata/colors.npy")

    # save your visualizations in results/
    tgt = "results"
    os.makedirs(tgt, exist_ok=True)

    # play with data before commenting out this.
    pdb.set_trace()

    colormapArray(data, colors, src, tgt)
