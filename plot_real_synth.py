

from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator, mute_direct
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import rmtree
import h5py as h5


def plot_two_gathers(data1, data2, pars):
    """
    Compares two shot gathers

    @params:

    @returns:
    """

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=[16, 8])


    clip = 0.1
    vmax = np.max(data1) * clip
    vmin = -vmax

    ax[0].imshow(data1,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')

    clip = 0.1
    vmax = np.max(data2) * clip
    vmin = -vmax

    ax[1].imshow(data2,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')
    plt.show()

def plot_two_traces(data1, data2, pars):
    """
        Compares two shot gathers
        
        @params:
        
        @returns:
        """
    
    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=[16, 8])
    
    
    clip = 0.1
    vmax = np.max(data1) * clip
    vmin = -vmax
    
    ax[0].plot(data1[:,1])
    ax[1].plot(data2[:,1])
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments to parse
    parser.add_argument("-f1", "--filename1",
                        type=str,
                        default="",
                        help="name of the file containing the synth data")
    parser.add_argument("-f2", "--filename2",
                        type=str,
                        default="",
                        help="name of the file containing the real data")

    # Parse the input
    args = parser.parse_args()


    def print_usage_error_message():
        print("\nUsage error.\n")
        parser.print_help()


    pars = ModelParameters()
    pars.dh = 6.25
    pars.peak_freq = 26
    pars.NX = 692 * 2
    pars.NZ = 752 * 2
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

    file = h5.File(args.filename1, "r")
    data1 = file['data'][:]
    vp = file['vp'][:]
    data1 = mute_direct(data1, vp[0], pars)
    file.close()

    file = h5.File(args.filename2, "r")
    data2 = file["data_cmp"][:data1.shape[0], 1:72]
    file.close()

    plot_two_gathers(data1, data2, pars)
    plot_two_traces(data1, data2, pars)





