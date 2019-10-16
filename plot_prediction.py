#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot the NN predictions
"""
from vrmslearn.Trainer import Trainer
from vrmslearn.SeismicGenerator import SeismicGenerator
from vrmslearn.RCNN import RCNN
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator, mute_direct, random_static, random_noise, mute_nearoffset, random_filt
from semblance.nmo_correction import semblance_gather, nmo_correction
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
import os
from shutil import rmtree
import h5py as h5
from scipy.signal import butter, lfilter
from scipy import ndimage, misc



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_predictions(modeled_data,
                     vp, vrms, vpred, tlabels, refpred, vint, vint_pred, pars):
    """
    This method creates one example by generating a random velocity model,
    modeling a shot record with it, and also computes the vrms. The three
    results are displayed side by side in an window.

    @params:

    @returns:
    """

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=[16, 8])

    im1 = ax[0].imshow(vp, cmap=plt.get_cmap('hot'), aspect='auto',
                       vmin=0.9 * pars.vp_min, vmax=1.1 * pars.vp_max)
    ax[0].set_xlabel("X Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_ylabel("Z Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_title("P Interval Velocity", fontsize=16, fontweight='bold')
    p = ax[0].get_position().get_points().flatten()
    axis_cbar = fig.add_axes([p[0], 0.03, p[2] - p[0], 0.02])
    plt.colorbar(im1, cax=axis_cbar, orientation='horizontal')

    clip = 0.05
    vmax = np.max(modeled_data) * clip
    vmin = -vmax

    ax[1].imshow(modeled_data,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')
    tlabels = [ii for ii, t in enumerate(tlabels) if t == 1]

    toff = np.zeros(len(tlabels)) + int(modeled_data.shape[1]/2)+1
    ax[1].plot(toff, tlabels, '*')
    refpred = [ii for ii, t in enumerate(refpred) if t == 1]
    toff = np.zeros(len(refpred)) + int(modeled_data.shape[1]/2)-2
    ax[1].plot(toff, refpred, 'r*')
    ax[1].set_xlabel("Receiver Index", fontsize=12, fontweight='normal')
    ax[1].set_ylabel("Time Index," + " dt = " + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax[1].set_title("Shot Gather", fontsize=16, fontweight='bold')

    ax[2].plot(vrms * (pars.vp_max-pars.vp_min) + pars.vp_min,
               np.arange(0, len(vrms)))
    ax[2].plot(vpred * (pars.vp_max - pars.vp_min) + pars.vp_min,
               np.arange(0, len(vpred)))
    ax[2].plot(vint * (pars.vp_max-pars.vp_min) + pars.vp_min,
               np.arange(0, len(vint)))
    ax[2].plot(vint_pred * (pars.vp_max - pars.vp_min) + pars.vp_min,
               np.arange(0, len(vint_pred)))
    ax[2].invert_yaxis()
    ax[2].set_ylim(top=0, bottom=len(vrms))
    ax[2].set_xlim(0.9 * pars.vp_min, 1.1 * pars.vp_max)
    ax[2].set_xlabel("RMS Velocity (m/s)", fontsize=12, fontweight='normal')
    ax[2].set_ylabel("Time Index," + " dt = " + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax[2].set_title("P RMS Velocity", fontsize=16, fontweight='bold')

    plt.show()


def plot_predictions_semb3(modeled_data,
                           vrms, vpred,
                           tlabels, refpred,
                           vint, vint_pred,
                           masks,
                           pars, dv=30, vmin=None, vmax = None,
                           clip=0.05, clipsemb=1.0,
                           plot_semb = True,
                           with_nmo = False,
                           textlabels = None,
                           savefile=None,
                           vint_pred_std=None,
                           vpred_std=None, tmin=None, tmax=None):
    """
    This method creates one example by generating a random velocity model,
    modeling a shot record with it, and also computes the vrms. The three
    results are displayed side by side in a window.

    @params:

    @returns:
    """

    NT = modeled_data[0].shape[0]
    ng = modeled_data[0].shape[1]
    dt = pars.resampling * pars.dt
    if vmin is None:
        vmin = pars.vp_min
    if vmax is None:
        vmax = pars.vp_max
    
    if pars.gmin ==-1 or pars.gmax ==-1:
        offsets = (np.arange(0, ng) - (ng) / 2) * pars.dh * pars.dg
    else:
        offsets = (np.arange(pars.gmin, pars.gmax, pars.dg)) * pars.dh

    times = np.reshape(np.arange(0, NT * dt, dt) - pars.tdelay, [-1])
    vels = np.arange(vmin - 5*dv, vmax + 2*dv, dv)

    if with_nmo:
        fig, ax = plt.subplots(3, 3, figsize=[11 / 2.54, 18 / 2.54])
    else:
        fig, ax = plt.subplots(3, 2, figsize=[8 / 2.54, 18 / 2.54])
    
    titles = [["a)", "b)", "c)"], ["d)", "e)", "f)"], ["g)", "h)", "i)"]]
    labels = ["True", "Pred", "Vint true", "Vint pred", "Vrms true", "Vrms pred", "Vrms std", "Vint std"]
    plots = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    for ii in range(3):
        if plot_semb:
            semb = semblance_gather(modeled_data[ii], times, offsets, vels)

        vmax = np.max(modeled_data[ii]) * clip
        vmin = -vmax
        ax[ii, 0].imshow(modeled_data[ii],
                         interpolation='bilinear',
                         cmap=plt.get_cmap('Greys'),
                         extent=[offsets[0] / 1000, offsets[-1] / 1000, times[-1], times[0]],
                         vmin=vmin, vmax=vmax,
                         aspect='auto')
        ymin, ymax = ax[ii, 0].get_ylim()
        if tmin is not None:
            if type(tmin) is list:
                ymax = tmin[ii]
            else:
                ymax = tmin
        if tmax is not None:
            if type(tmax) is list:
                ymin = tmax[ii]
            else:
                ymin = tmax
        xmin, xmax = ax[ii, 0].get_xlim()
        if tlabels is not None:
            tlabels[ii] = [jj * dt - pars.tdelay for jj, t in enumerate(tlabels[ii]) if t == 1]
        refpred[ii] = [jj * dt - pars.tdelay for jj, t in enumerate(refpred[ii]) if t == 1]
        if np.min(offsets) < 0:
            if tlabels is not None:
                tofflabels = np.zeros(len(tlabels[ii])) - 2 * pars.dh * pars.dg
            toffpreds = np.zeros(len(refpred[ii])) + 2 * pars.dh * pars.dg
        else:
            if tlabels is not None:
                tofflabels = np.zeros(len(tlabels[ii])) + np.min(np.abs(offsets)) + 1 * pars.dh * pars.dg
            toffpreds = np.zeros(len(refpred[ii])) + np.min(np.abs(offsets)) + 3 * pars.dh * pars.dg
        if tlabels is not None:
            plots[0], = ax[ii, 0].plot(tofflabels / 1000, tlabels[ii], 'r*', markersize=3)
        plots[1], = ax[ii, 0].plot(toffpreds / 1000, refpred[ii], 'b*', markersize=3)

        ax[ii, 0].set_xlabel("Offset (km)")
        ax[ii, 0].set_ylabel("Time (s)")
        #ax[ii, 0].set_title(titles[0][0])


        ax[ii, 0].text(xmin - 0.3 * (xmax-xmin), ymax + 0.1*(ymax-ymin),
                       titles[0][ii], fontsize="large")

        # ax[ii, 2 * jj].xaxis.set_ticks(np.arange(-1, 1.5, 0.5))

        if ii == 0:
            ax[ii, 0].legend(plots[0:2], labels[0:2], loc='upper right',
                                  bbox_to_anchor=(1.13, 1.29))
        if plot_semb:
            vmax = np.max(semb) * clipsemb
            vmin = np.min(semb)
            ax[ii, 1].imshow(semb,
                             extent=[(vels[0] - dv / 2) / 1000,
                             (vels[-1] - dv / 2) / 1000, times[-1], times[0]],
                             cmap=plt.get_cmap('YlOrRd'),
                             vmin=vmin, vmax=vmax,
                             interpolation='bilinear',
                             aspect='auto')
        if masks is not None:
            if vint is not None:
                vint[ii][masks[ii] == 0] = np.NaN
            if vrms is not None:
                vrms[ii][masks[ii] == 0] = np.NaN
            vint_pred[ii][masks[ii] == 0] = np.NaN
            vpred[ii][masks[ii] == 0] = np.NaN
        if vint is not None:
            plots[2], = ax[ii, 1].plot(vint[ii] / 1000, times, '-', color='lightgray')
        if vint_pred_std is not None:
            plots[6], = ax[ii, 1].plot((vint_pred[ii] + vint_pred_std[ii]) / 1000, times, '-', color='lightgreen', alpha=0.4)
            ax[ii, 1].plot((vint_pred[ii] - vint_pred_std[ii]) / 1000, times, '-', color='lightgreen', alpha=0.4)
        if vrms is not None:
            plots[4], = ax[ii, 1].plot(vrms[ii] / 1000, times, '-g', color='black')
        plots[5], = ax[ii, 1].plot(vpred[ii] / 1000, times, '-b')
        plots[3], = ax[ii, 1].plot(vint_pred[ii] / 1000, times, '-', color='lightgreen')

        if vpred_std is not None:
            plots[7], = ax[ii, 1].plot((vpred[ii] + vpred_std[ii]) / 1000, times,  '-b', alpha=0.2)
            ax[ii, 1].plot((vpred[ii] - vpred_std[ii]) / 1000, times, '-b', alpha=0.2)


        ax[ii, 1].xaxis.set_ticks(np.arange(np.ceil(np.min(vels)/1000),
                                            1+np.floor(np.max(vels)/1000)))

        ax[ii, 1].set_ylim(bottom=ymin, top=ymax)
        ax[ii, 0].set_ylim(bottom=ymin, top=ymax)
        xmin, xmax = ax[ii, 1].get_xlim()
        ax[ii, 1].set_xlabel("Velocity (km/s)")
        ax[ii, 1].set_ylabel("Time (s)")
        ax[ii, 1].text(xmin - 0.3 * (xmax - xmin), ymax + 0.1 * (ymax - ymin),
                       titles[1][ii], fontsize="large")
        if textlabels:
            ax[ii, 1].text(xmin + 0.94 * (xmax - xmin), ymax + - 0.03 * (ymax - ymin),
                           textlabels[ii],  ha="right", va="top", fontsize="large")

        if ii == 0:
            ax[ii, 1].legend(plots[2:6], labels[2:6],
                         loc='upper right',
                         bbox_to_anchor=(1.15, 1.50),
                         handlelength=0.4)
        if with_nmo:
            vmax = np.max(modeled_data[ii]) * clip
            vmin = -vmax
            data_nmo = nmo_correction(modeled_data[ii], times, offsets, vpred[ii], stretch_mute=0.3)
            ax[ii, 2].imshow(data_nmo,
                             interpolation='bilinear',
                             cmap=plt.get_cmap('Greys'),
                             extent=[offsets[0] / 1000, offsets[-1] / 1000, times[-1], times[0]],
                             vmin=vmin, vmax=vmax,
                             aspect='auto')
            ax[ii, 2].set_ylim(bottom=ymin, top=ymax)
            ax[ii, 2].set_xlabel("Offset (km)")
            ax[ii, 2].set_ylabel("Time (s)")
            xmin, xmax = ax[ii, 0].get_xlim()
            ax[ii, 2].text(xmin - 0.3 * (xmax-xmin), ymax + 0.1*(ymax-ymin),
                           titles[2][ii], fontsize="large")

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    if savefile:
        plt.savefig(savefile, dpi=600)
        plt.savefig(savefile+"_lowres", dpi=100)
    plt.show()




if __name__ == "__main__":

    # Set pref_device_type = 4
    pref_device_type = 4

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="name of the directory to save logs : str"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dataset_1/dhmin40_layer_num_min5/example_1_31891",
        help="name of the directory to save logs : str"
    )
    parser.add_argument(
        "--fileparam",
        type=str,
        default="dataset_1/dhmin40_layer_num_min5/example_1_31891",
        help="name of the directory that contains the model parameters: str"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=5000,
        help="number of training iterations : int > 0"
    )
    parser.add_argument(
        "--nbatch",
        type=int,
        default=10,
        help="number of gathers in one batch : int > 0"
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        default=2,
        help="number of layers in the model : int > 0"
    )
    parser.add_argument(
        "--layer_num_min",
        type=int,
        default=5,
        help="number of layers in the model : int > 0"
    )
    parser.add_argument("-d", "--device",
                        type=int,
                        default=4,
                        help="device type : int = 2 or 4, default = 2")


    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    # Test for input errors
    def print_usage_error_message():
        print("\nUsage error.\n")
        parser.print_help()

    if args.niter < 0:
        print_usage_error_message()
        exit()

    if args.nlayers <= -1:
        print_usage_error_message()
        exit()

    if args.nbatch <= 0:
        print_usage_error_message()
        exit()

    parameters = ModelParameters()
    parameters.read_parameters_from_disk(args.fileparam)
    parameters.device_type = args.device
    parameters.num_layers = args.nlayers
    #parameters.read_parameters_from_disk(filename='dataset_3/dhmin40_layer_num_min5/model_parameters.hdf5')
    gen = SeismicGenerator(parameters)

    parameters.mute_nearoffset = False
    parameters.random_static = False
    parameters.random_noise = False
    data, vrms, vint, valid, tlabels = gen.read_example(".", filename=args.filename)


#    data = mute_direct(data, 1500, parameters)
#    #data = random_static(data, 2)
##    data = random_noise(data, 0.01)
##    data = mute_nearoffset(data, 10)
##    data = random_filt(data, 9)
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    vrms = np.expand_dims(vrms, axis=0)
    vint = np.expand_dims(vint, axis=0)
    valid = np.expand_dims(valid, axis=0)
    tlabels = np.expand_dims(tlabels, axis=0)
    f = h5.File(args.filename, "r")
    vp = f['vp'][:]
    f.close()



    nn = RCNN(input_size=gen.image_size,
              batch_size=1)
    trainer = Trainer(NN=nn,
                      data_generator=gen,
                      totrain=False)

    preds = trainer.evaluate(toeval=[nn.output_ref, nn.output_vint, nn.output_vrms],
                             niter=args.niter,
                             dir=args.logdir,
                             batch=[data, vrms, vint, valid, tlabels])

    refpred = np.argmax(preds[0][0,:], axis=1)
    vint_pred = preds[1]
    vpred = preds[2]
    vp = np.stack([vp] * vp.shape[0], axis=1)


    plot_predictions_semb(data[0,:,:,0],
                          vp,
                          vrms[0,:],
                          vpred[0,:],
                          tlabels[0,:],
                          refpred, vint[0,:], vint_pred[0,:], parameters, with_semb=False)

