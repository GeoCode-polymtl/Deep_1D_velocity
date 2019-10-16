#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot one example with generated data
"""

from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator, mute_direct, random_time_scaling, random_noise, random_static
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import rmtree
import h5py as h5

def plot_one_example(modeled_data, vrms, vp, tlabels, pars):
    """
    This method creates one example by generating a random velocity model,
    modeling a shot record with it, and also computes the vrms. The three
    results are displayed side by side in an window.

    @params:

    @returns:
    """

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=[16, 8])

    im1 = ax[0].imshow(vp, cmap=plt.get_cmap('hot'), aspect='auto', vmin=0.9 * pars.vp_min, vmax=1.1 * pars.vp_max)
    ax[0].set_xlabel("X Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_ylabel("Z Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_title("P Interval Velocity", fontsize=16, fontweight='bold')
    p = ax[0].get_position().get_points().flatten()
    axis_cbar = fig.add_axes([p[0], 0.03, p[2] - p[0], 0.02])
    plt.colorbar(im1, cax=axis_cbar, orientation='horizontal')

    clip = 0.1
    vmax = np.max(modeled_data) * clip
    vmin = -vmax

    ax[1].imshow(modeled_data,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')

    refpred = [ii for ii, t in enumerate(tlabels) if t == 1]
    if pars.minoffset == 0:
        toff = np.zeros(len(refpred)) + int(modeled_data.shape[1]/2)-2
    else:
        toff = np.zeros(len(refpred))
    ax[1].plot(toff, refpred, 'r*')

    ax[1].set_xlabel("Receiver Index", fontsize=12, fontweight='normal')
    ax[1].set_ylabel("Time Index," + " dt = " + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax[1].set_title("Shot Gather", fontsize=16, fontweight='bold')

    ax[2].plot(vrms * (pars.vp_max-pars.vp_min) + pars.vp_min, np.arange(0, len(vrms)))
    ax[2].invert_yaxis()
    ax[2].set_ylim(top=0, bottom=len(vrms))
    ax[2].set_xlim(0.9 * pars.vp_min, 1.1 * pars.vp_max)
    ax[2].set_xlabel("RMS Velocity (m/s)", fontsize=12, fontweight='normal')
    ax[2].set_ylabel("Time Index," + " dt = " + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax[2].set_title("P RMS Velocity", fontsize=16, fontweight='bold')

    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Add arguments to parse
    parser.add_argument("-l", "--nlayers",
                        type=int,
                        default=12,
                        help="number of layers : int > 0, default = 0")
    parser.add_argument("-d", "--device",
                        type=int,
                        default=4,
                        help="device type : int = 2 or 4, default = 2")
    parser.add_argument("-f", "--filename",
                        type=str,
                        default="",
                        help="name of the file containing the example")

    # Parse the input
    args = parser.parse_args()

    pars = ModelParameters()
    pars.dh = 6.25
    pars.peak_freq = 26
    pars.NX = 692*2
    pars.NZ = 752*2
    pars.dt = 0.0004
    pars.NT = int(8.0 / pars.dt)
    pars.resampling = 10
    
    pars.dg = 8
    pars.gmin = int(470 / pars.dh)
    pars.gmax = int((470 + 72 * pars.dg * pars.dh) / pars.dh)
    pars.minoffset = 470
    
    pars.vp_min = 1300.0  # maximum value of vp (in m/s)
    pars.vp_max = 4000.0  # minimum value of vp (in m/s)
    
    pars.marine = True
    pars.velwater = 1500
    pars.d_velwater = 60
    pars.water_depth = 3500
    pars.dwater_depth = 1000
    
    pars.fs = False
    pars.source_depth = (pars.Npad + 4) * pars.dh
    pars.receiver_depth = (pars.Npad + 4) * pars.dh
    pars.identify_direct = False
    
    pars.random_time_scaling = True
    
    gen = SeismicGenerator(pars)
    if args.filename is "":
        workdir = "./seiscl_workdir"
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
        data, vrms, vp, valid, tlabels = gen.compute_example(workdir=workdir)
        if os.path.isdir(workdir):
            rmtree(workdir)
    else:
        file = h5.File(args.filename, "r")
        data = file['data'][:]
        vrms = file['vrms'][:]
        vp = file['vp'][:]
        valid = file['valid'][:]
        tlabels = file['tlabels'][:]
        file.close()
    
    vp = np.stack([vp] * vp.shape[0], axis=1)
    data = mute_direct(data, vp[0, 0], pars)
    data = random_time_scaling(data, pars.dt * pars.resampling, emin=-2, emax=2)
    data = random_noise(data, 0.02)
    random_static(data, 2)
    plot_one_example(data, vrms, vp, tlabels, pars)

