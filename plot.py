import numpy as np
import matplotlib.pyplot as plt

import glob
from PIL import Image
from utils import create_dataset

import os
import argparse

import torch


def multiple_grid(grids, vmin_sim, vmax_sim,
                  vmin_err, vmax_err,
                  save=None, show=False):
    """
    Procedure to save a colormap of the temperature
    args:
    -----
        grids: a list of three 2D grid of temperature with size [H, W]
        vmin: Minimum Temperature of the simulation
        vmax: Maximum Temperature of the simulation
        save: string if not None save the picture to save
        show: boolean whether or not displaying the plot
    """

    fig, axes = plt.subplots(4, 1)
    fig.set_figheight(8)
    fig.set_figwidth(17)

    images = []
    ylabels = ["Target", "Prediction", "Absolute error"]
    for i, ax in enumerate(axes[:-1]):
        if i == 2:
            vmin, vmax = vmin_err, vmax_err

        else:
            vmin, vmax = vmin_sim, vmax_sim

        im = ax.imshow(np.flip(grids[i], 0), cmap=plt.cm.RdBu_r, aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='bilinear',
                       origin='lower')

        ax.set_ylabel(ylabels[i])

        images.append(im)
        # ax.set_ylim((0, 10))

    # ax.set_xlabel("Width")
    # axes[1].set_ylabel("Height")

    # Create colorbar for simulations
    bbox_ax0 = axes[0].get_position()
    bbox_ax1 = axes[1].get_position()
    bbox_ax2 = axes[2].get_position()
    left_ax0, bottom_ax0, width_ax0, height_ax0 = bbox_ax0.bounds
    left_ax1, bottom_ax1, width_ax1, height_ax1 = bbox_ax1.bounds
    left_ax2, bottom_ax2, width_ax2, height_ax2 = bbox_ax2.bounds

    left_cmsim = left_ax1 + width_ax1 + 0.01
    bottom_cmsim = bottom_ax1
    width_cmsim = 0.05
    height_cmsim = bottom_ax0 - bottom_ax1 + height_ax0

    left_cmerr = left_ax2 + width_ax2 + 0.01
    bottom_cmerr = bottom_ax2
    width_cmerr = 0.05
    height_cmerr = height_ax2

    ax_cmsim = fig.add_axes(
        [left_cmsim, bottom_cmsim, width_cmsim, height_cmsim])
    fig.colorbar(images[0], cax=ax_cmsim)

    ax_cmerr = fig.add_axes(
        [left_cmerr, bottom_cmerr, width_cmerr, height_cmerr])
    fig.colorbar(images[2], cax=ax_cmerr)

    bar_ax = axes[-1]
    bar_ax.barh([0], np.max(grids[-1]))
    bar_ax.set_xlim((0, vmax_err))
    bar_ax.set_yticks([])
    bar_ax.set_ylabel("Max absolute error")

    fig.align_ylabels()

    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()

def make_gif(fp_in, fp_out):
    """
    fp_in: string path to the images for the creation of the gif following bash command
           e.g.: "imgs/image_*.png"
    fp_out: string filenale where to save the gif: e.g. "image.gif"
    """

    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in),
            key=lambda x: int(x.split("_")[1].split(".")[0])))

    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=1, loop=0)

