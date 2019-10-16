#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Performs the testing on the real dataset (reproduces Figures 5 and 6)
"""

from plot_prediction import plot_predictions_semb3
from semblance.nmo_correction import stack
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import mode
from scipy.signal import medfilt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
import segyio

from vrmslearn.Trainer import Trainer
from vrmslearn.RCNN import RCNN
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.ModelGenerator import generate_random_2Dlayered, interval_velocity_time, calculate_vrms
from vrmslearn.SeismicGenerator import SeismicGenerator, mute_direct, random_static
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import rmtree
import h5py as h5
import tensorflow as tf
import fnmatch
from scipy.signal import medfilt
import argparse
import time

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if lowcut==0:
        b, a = butter(order, high, btype='lowpass', analog=False)
    elif highcut==0:
        b, a = butter(order, low, btype='highpass', analog=False)
    else:
        b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def bandpass(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots",
                        type=int,
                        default=1,
                        help="1: plot only the first CMP results "
                             "2: plot the2D RMS and interval velocity section and the "
                             "stacked section"
                        )
    parser.add_argument("--logdir",
                        type=str,
                        default="Case4b2/2_schedule2_lr0.000800_eps_0.000010_beta10.900000_beta20.980000_batch_size_40",
                        help="name of the directory of the checkpoint: str"
                        )
    parser.add_argument(
                        "--niter",
                        type=int,
                        default=10000,
                        help="number of training iterations of the checkpoint"
                        )
    parser.add_argument(
                        "--savepred",
                        type=int,
                        default=1,
                        help="Save predictions to a file. 0: no, 1: yes"
                        )
    parser.add_argument(
                        "--recompute",
                        type=int,
                        default=0,
                        help="Recompute predictions. 0: no, 1: yes"
                        )
                        
    # Parse the input
    args = parser.parse_args()
    
    dirs = []
    for dir1 in os.listdir('./'):
        if os.path.isdir(dir1):
            for dir2 in os.listdir(dir1):
                path2 = os.path.join(dir1, dir2)
                if os.path.isdir(path2):
                    dirs.append(path2)

    logdirs = fnmatch.filter(dirs, args.logdir)
    print(logdirs)

    create_data = True
    logdir = args.logdir
    niter = args.niter
    max_batch = 100

    pars = ModelParameters()
    pars.layer_dh_min = 5
    pars.layer_num_min = 48

    pars.dh = 6.25
    pars.peak_freq = 26
    pars.df = 5
    pars.wavefuns = [0, 1]
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
    
    pars.tdelay*=1.5
    padt = int(pars.tdelay / pars.dt / pars.resampling) * 0

    savefile = "./realdata/survey.hdf5"
    ng = 72

    file = h5.File(savefile, "r")
    data_cmp = file["data_cmp"]

    nbatch = int(data_cmp.shape[1] / ng / max_batch)
    ns = int(data_cmp.shape[1] / ng)

    nn = RCNN([data_cmp.shape[0]+padt, ng],
              batch_size=max_batch, use_peepholes=False)

    if args.plots ==1:
        nbatch = 1
        ns = max_batch

    refpred = []
    vint_pred = []
    vpred = []

    for logdir in logdirs:
        if not os.path.isfile(logdir + '/realdatapred.h5') or args.recompute:
            print('recomputing')
            if os.path.isfile(logdir + '/model.ckpt-' + str(niter) + '.meta'):
                data =  np.zeros([max_batch, data_cmp.shape[0] + padt, ng, 1])
                refpred.append(np.zeros([data_cmp.shape[0]+ padt, ns]))
                vint_pred.append(np.zeros([data_cmp.shape[0]+ padt, ns]))
                vpred.append(np.zeros([data_cmp.shape[0]+ padt, ns]))
                with nn.graph.as_default():
                    saver = tf.train.Saver()
                    with tf.Session() as sess:
                        saver.restore(sess, logdir + '/model.ckpt-' + str(niter))
                        start_time = time.time()
                        for ii in range(nbatch):
                            for jj in range(max_batch):
                                idmin = ii * max_batch * ng + ng * jj
                                idmax = ii * max_batch * ng + ng * (jj + 1)
                                data[jj, padt:, :, 0] = data_cmp[:, idmin:idmax]
                            evaluated = sess.run([nn.input_scaled, nn.output_ref,
                                                  nn.output_vint, nn.output_vrms],
                                                 feed_dict={nn.input: data})
                            idmin = ii*max_batch
                            idmax = (ii+1)*max_batch
                            refpred[-1][:, idmin:idmax] = np.transpose(np.argmax(evaluated[1], axis=2), [1, 0])
                            vint_pred[-1][:, idmin:idmax] = np.transpose(evaluated[2])
                            vpred[-1][:, idmin:idmax] = np.transpose(evaluated[3])
                        print("--- %s seconds ---" % (time.time() - start_time))
                if args.savepred==1:
                    filesave = h5.File(logdir + '/realdatapred.h5', "w")
                    filesave['refpred'] = refpred[-1]
                    filesave['vint_pred'] = vint_pred[-1]
                    filesave['vpred'] = vpred[-1]
                    filesave.close()
        else:
            filesave = h5.File(logdir + '/realdatapred.h5', "r")
            refpred.append(filesave['refpred'][:])
            vint_pred.append(filesave['vint_pred'][:])
            vpred.append(filesave['vpred'][:])
            filesave.close()
    t = np.arange(0, data_cmp.shape[0]) * pars.dt*pars.resampling - pars.tdelay
    offsets = (np.arange(pars.gmin, pars.gmax, pars.dg)) * pars.dh
    vrms = gaussian_filter(np.mean(vpred, axis=0), [1, 9]) * (pars.vp_max - pars.vp_min) + pars.vp_min
    vint = gaussian_filter(np.median(vint_pred, axis=0), [1, 9]) * (pars.vp_max - pars.vp_min) + pars.vp_min
    
    if not os.path.isfile("./realdata/survey_stacked.hdf5") or (args.recompute and args.plots==2):
       
        stacked = np.zeros([data_cmp.shape[0], ns])
        for ii in range(ns):
            stacked[:, ii] = stack(data_cmp[:, ii*ng:(ii+1)*ng],
                                   t, offsets, vrms[:,ii])
        filesave = h5.File("./realdata/survey_stacked.hdf5", "w")
        filesave['stacked'] = stacked
        filesave.close()
    else:
        filesave = h5.File("./realdata/survey_stacked.hdf5", "r")
        stacked = filesave['stacked'][:]
        filesave.close()

    if args.plots == 1:
        shots = [250, 1000, 1750]
        datas = [data_cmp[:, ii*ng:(ii+1)*ng] for ii in shots]
        vrmss = [np.mean([v[:,ii] for v in vpred], axis=0) for ii in shots]
        vrmss = [v * (pars.vp_max - pars.vp_min) + pars.vp_min for v in vrmss]
        vints = [np.mean([v[:,ii] for v in vint_pred], axis=0) for ii in shots]
        vints = [v * (pars.vp_max - pars.vp_min) + pars.vp_min for v in vints]
        refs = [mode([v[:,ii] for v in refpred] , axis=0).mode[0] for ii in shots]
        vrms_stds = [np.std([v[:,ii]* (pars.vp_max - pars.vp_min) + pars.vp_min
                             for v in vpred], axis=0) for ii in shots]
        vint_stds = [np.std([v[:,ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
                             for v in vint_pred], axis=0) for ii in shots]

        plot_predictions_semb3(datas,
                              None,
                              vrmss,
                              None,
                              refs,
                              None,
                              vints, None,
                              pars, plot_semb=True, vmin=1400, vmax=3400, dv=50,
                               vpred_std =vrms_stds,
                               vint_pred_std = vint_stds, clip=0.05,
                               tmin = 2, tmax=10,
                               savefile="./Paper/Fig/realdata_semblance",
                               with_nmo=True
                               )


    if args.plots == 2:
        
        def plot_model(thisax, v, label, extent = None, cbar=True, vmin=None, vmax=None,
                       cmap=None):
            if cmap is None:
                cmap=plt.get_cmap("jet")
            im = thisax.imshow(v, cmap=cmap,
                               interpolation='bilinear',
                               aspect="auto",
                               extent=extent, vmin=vmin, vmax=vmax)
            thisax.set_xlabel('CMP')
            thisax.set_ylabel('T (s)')
            thisax.set_ylim(bottom=10, top=2)
            thisax.set_xlim(left=1, right=2080)
            divider = make_axes_locatable(thisax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            if cbar:
                clr = plt.colorbar(im, cax=cax)
                cax.xaxis.set_ticks_position("top")
                cax.xaxis.tick_top()
                cax.set_xlabel('V (km/s)', labelpad=10)
                cax.xaxis.set_label_position('top')
            else:
                 cax.axis('off')
            ymin, ymax =thisax.get_ylim()
            xmin, xmax = thisax.get_xlim()
            thisax.text(xmin - 0.05 * (xmax - xmin), ymax + 0.15 * (ymax - ymin),
                        label,  ha="right", va="top", fontsize="large")

        fig = plt.figure(figsize=(15 / 2.54, 23 / 2.54))
        gridspec.GridSpec(4,1)


        extent = [0, vrms.shape[1], np.max(t), 0]
        plot_model(plt.subplot2grid( (4,1), (0,0)), vrms/1000, "a)", extent=extent)
        plot_model(plt.subplot2grid( (4,1), (1,0)), vint/1000, "b)", extent=extent, vmin=1.4, vmax=3.1)
        clip = 0.15
        stacked = stacked * (np.reshape(t, [-1, 1])**2  + 1e-6)
        stacked = stacked / np.sqrt(np.sum(stacked**2, axis=0))
        vmax = np.max(stacked) * clip
        vmin = -vmax
        plot_model(plt.subplot2grid( (4,1), (2,0)), stacked, "c)", extent=extent, cbar=False, cmap=plt.get_cmap('Greys'), vmax=vmax, vmin=vmin)
        
        NT = stacked.shape[0]
        with segyio.open("./realdata/USGS_line32/CSDS32_1.SGY", "r",
                         ignore_geometry=True) as segy:
            stacked_usgs = np.transpose(np.array([segy.trace[trid]
                                                  for trid in range(segy.tracecount)]))
        stacked_usgs = stacked_usgs[:, -2401:-160]
        stacked_usgs = stacked_usgs[:,::-1]
        for kk in range(stacked_usgs.shape[1]):
            stacked_usgs[:, kk] = stacked_usgs[:, kk] / np.sqrt(np.sum(stacked_usgs[:, kk] **2)+1e-4)
        clip = 0.25
        vmax = np.max(stacked_usgs) * clip
        vmin = -vmax
        plot_model(plt.subplot2grid( (4,1), (3,0)), stacked_usgs, "d)", extent=extent, cbar=False, cmap=plt.get_cmap('Greys'), vmax=vmax, vmin=vmin)
        
        plt.tight_layout()#rect=[0, 0, 1, 0.995])
        plt.savefig("./Paper/Fig/realdata_stacked", dpi=600)
        plt.savefig("./Paper/Fig/realdata_stacked_lowres", dpi=100)
        plt.show()

    file.close()
