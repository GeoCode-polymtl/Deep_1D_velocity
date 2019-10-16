#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Defines parameters for 2D testing, creates the dataset make predictions
"""
from vrmslearn.Trainer import Trainer
from vrmslearn.RCNN import RCNN
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.ModelGenerator import generate_random_2Dlayered, interval_velocity_time, calculate_vrms
from vrmslearn.SeismicGenerator import SeismicGenerator, mute_direct
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from shutil import rmtree
import h5py as h5
import fnmatch
from scipy.signal import medfilt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

def cmp_pos(rec_pos, src_pos, bin):
    ng = rec_pos.shape[0] / src_pos.shape[0]
    
    src_pos = np.repeat(src_pos, ng)
    cmps = ((src_pos + rec_pos) / 2 / bin).astype(int) * bin
    offsets = src_pos - rec_pos
    
    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    unique_cmps, counts = np.unique(cmps, return_counts=True)
    cmax = np.max(counts)
    firstcmp = unique_cmps[np.argmax(counts == cmax)]
    lastcmp = unique_cmps[-np.argmax(counts[::-1] == cmax) - 1]
    ind1 = np.argmax(cmps == firstcmp)
    ind2 = np.argmax(cmps > lastcmp)
    
   
    return (ind2-ind1)/cmax

def sort_cmps(data, rec_pos, src_pos, bin):

    ng = rec_pos.shape[0] / src_pos.shape[0]

    src_pos = np.repeat(src_pos, ng)
    cmps = ((src_pos + rec_pos) / 2 / bin).astype(int) * bin
    offsets = src_pos - rec_pos

    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    unique_cmps, counts = np.unique(cmps, return_counts=True)
    cmax = np.max(counts)
    firstcmp = unique_cmps[np.argmax(counts == cmax)]
    lastcmp = unique_cmps[-np.argmax(counts[::-1] == cmax) - 1]
    ind1 = np.argmax(cmps == firstcmp)
    ind2 = np.argmax(cmps > lastcmp)
    ntraces = cmps[ind1:ind2].shape[0]
    data_cmp = np.zeros([data.shape[0], ntraces])
    n = 0
    for ii, jj in enumerate(ind):
        if ii >= ind1 and ii < ind2:
            data_cmp[:, n] = data[:, jj]
            n += 1
    return data_cmp

def get_first_cmp_pos(rec_pos, src_pos, bin):
    ng = rec_pos.shape[0] / src_pos.shape[0]
    src_pos = np.repeat(src_pos, ng)
    cmps = ((src_pos + rec_pos) / 2 / bin).astype(int) * bin
    offsets = src_pos - rec_pos

    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    unique_cmps, counts = np.unique(cmps, return_counts=True)
    cmax = np.max(counts)
    firstcmp = unique_cmps[np.argmax(counts == cmax)]

    return firstcmp


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    
    # Add arguments to parse for training
    parser.add_argument(
                        "--logdir",
                        type=str,
                        default="logs",
                        help="Checkpoint filename for which to predict"
                        )
    parser.add_argument(
                        "--niter",
                        type=int,
                        default=1000,
                        help="Iteration number of the checkpoint file"
                        )
    parser.add_argument(
                        "--create_data",
                        type=int,
                        default=1,
                        help="If 1: create the 2D dataset"
                        )
    parser.add_argument(
                        "--data_from",
                        type=int,
                        default=0,
                        help="Start that example creation from data_from"
                        )
                        
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    ndatasets = 100
    
    """
        __________________Find model directories______________________
    """
    dirs = []
    for dir1 in os.listdir('./'):
        if os.path.isdir(dir1):
            for dir2 in os.listdir(dir1):
                path2 = os.path.join(dir1, dir2)
                if os.path.isdir(path2):
                    dirs.append(path2)

    logdirs = fnmatch.filter(dirs, args.logdir)
    logdirs.sort()

    """
    _________________________Define the parameters______________________
    """
    pars = ModelParameters()
    pars.flat = False
    pars.NX = 1700
    pars.NZ = 750 * 2
    pars.dh = 6.25
    pars.peak_freq = 26

    pars.num_layers = 0
    pars.layer_dh_min = 10  # minimum number of grid cells that a layer must span
    pars.layer_num_min = 25  # minimum number of layers
    pars.angle_max = 8
    pars.dangle_max = 3
    pars.amp_max = 0
    pars.max_texture = 0.08
    pars.texture_xrange = 1
    pars.texture_zrange = 1.95*pars.NZ

    pars.vp_min = 1300.0  # maximum value of vp (in m/s)
    pars.vp_max = 4000.0  # minimum value of vp (in m/s)

    pars.dt = 0.0004
    pars.resampling = 10
    pars.NT = int(8.0 / pars.dt)
    pars.marine = True
    pars.velwater = 1500
    pars.d_velwater = 60
    pars.water_depth = 3500
    pars.dwater_depth = 1000
    pars.dg = 8
    pars.gmin = int(470 / pars.dh)
    pars.gmax = int((470 + 72 * pars.dg * pars.dh) / pars.dh)
    pars.minoffset = 470

    pars.fs = False
    pars.source_depth = (pars.Npad + 4) * pars.dh
    pars.receiver_depth = (pars.Npad + 4) * pars.dh
    pars.identify_direct = False
    
    pars.mute_dir = True

    gen = SeismicGenerator(model_parameters=pars)
    ds = pars.dg
    ng = 72
    dg = pars.dg
    nearoffset = int(pars.minoffset / pars.dh)
    length = ng * dg + nearoffset

    sx = np.arange(pars.Npad + length + 1,
                   pars.NX - pars.Npad - length,
                   ds) * pars.dh
    sz = sx * 0 + pars.source_depth
    sid = np.arange(0, sx.shape[0])
    gen.F.src_pos = np.stack([sx,
                               sx * 0,
                               sz,
                               sid,
                               sx * 0 + pars.sourcetype], axis=0)
    gen.F.src_pos_all = gen.F.src_pos
    gen.F.src = np.empty((gen.F.csts['NT'], 0))

    gx = np.concatenate([ s - np.arange(nearoffset, length, dg) * pars.dh for s in sx], axis=0)
    gz = gx * 0 + pars.receiver_depth
    gid = np.arange(0, len(gx))
    gsid = np.repeat(sid, ng)
    gen.F.rec_pos = np.stack([gx,
                               gx * 0,
                               gz,
                               gsid,
                               gid,
                               gx * 0 + 2,
                               gx * 0,
                               gx * 0], axis=0)
    gen.F.rec_pos_all = gen.F.rec_pos

    ncmps = cmp_pos(gen.F.rec_pos[0,:], gen.F.src_pos[0,:], ds * pars.dh)
    
    """
        _________________________Generate the dataset______________________
    """

    workdir = "seiscl_workdir"
    savedir = "dataset_article/test2D"
    if not os.path.isdir(workdir):
        os.mkdir(workdir)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    examples = fnmatch.filter(os.listdir(savedir), 'example_*')
    if args.create_data:
        pars.save_parameters_to_disk(savedir + "/model_parameters.hdf5")
        for ii in range(args.data_from, ndatasets):
            savefile = "example_%d" % ii
            if savefile not in examples:

                vp, vs, rho, vels, layers, angles = generate_random_2Dlayered(pars, seed=ii)
                file = h5.File(savedir + "/" + savefile, "a")
                file["vp"] = vp
                file["vels"] = vels
                file["layers"] = layers
                file["angles"] = angles
#                cmp0 = get_first_cmp_pos(gen.F.rec_pos[0,:], gen.F.src_pos[0,:], ds * pars.dh)
#                ind0 = int(cmp0 / pars.dh)
#                indm = int(ind0+ncmps)
#
#                plt.imshow(vp[:, ind0:indm], cmap=plt.get_cmap("jet"), aspect="auto", interpolation='bilinear')
#                plt.colorbar()
#                plt.show()

                gen.F.set_forward(gen.F.src_pos[3, :],
                                  {'vp': vp, 'vs': vs, 'rho': rho},
                                  workdir,
                                  withgrad=False)
                gen.F.execute(workdir)
                data = gen.F.read_data(workdir)[0]
                file["data"] = data
                data_cmp = sort_cmps(data,
                                     gen.F.rec_pos[0,:],
                                     gen.F.src_pos[0,:],
                                     ds * pars.dh)
                file["data_cmp"] = data_cmp
                file.close()

    rmtree(workdir)

    """
    ______________________Make predictions for each model______________________
    """

    examples = fnmatch.filter(os.listdir(savedir), 'example_*')
    for logdir in logdirs:
        preddir = os.path.join(savedir, logdir)
        if not os.path.isdir(preddir):
            os.makedirs(preddir)
        predictions = fnmatch.filter(os.listdir(preddir), 'example_*_pred')
        for ii in range(ndatasets):
            savefile = "example_%d" % ii
            if savefile in examples and savefile + "_pred" not in predictions:
                print(preddir)
                print(savefile)
                file = h5.File(savedir + "/" + savefile, "r")
                data_cmp = file["data_cmp"][::pars.resampling,:]
                vp = file["vp"][:]
                file.close()

                ns = int(data_cmp.shape[1] / ng)
                data = np.zeros([ns, data_cmp.shape[0], ng, 1])
                for jj in range(ns):
                    data[jj, :, :, 0] = mute_direct(data_cmp[:, ng * jj:ng * (jj + 1)], vp[0,0], pars)


                vrms = np.zeros([data.shape[0], data.shape[1]])
                vint = np.zeros([data.shape[0], data.shape[1]])
                vint = np.zeros([data.shape[0], data.shape[1]])
                valid = np.zeros([data.shape[0], data.shape[1]])
                tlabels = np.zeros([data.shape[0], data.shape[1]])

                nn = RCNN(input_size=data[0,:,:,0].shape,
                          batch_size=ns)
                trainer = Trainer(NN=nn,
                                  data_generator=gen,
                                  totrain=False)

                preds = trainer.evaluate(toeval=[nn.output_ref,
                                                 nn.output_vint,
                                                 nn.output_vrms],
                                         niter=args.niter,
                                         dir=logdir,
                                         batch=[data, vrms, vint, valid, tlabels])
                refpred = np.argmax(preds[0], axis=2)
                vint_pred = preds[1]
                vrms_pred = preds[2]

                vint_pred = vint_pred * (pars.vp_max - pars.vp_min) + pars.vp_min
                vrms_pred = vrms_pred * (pars.vp_max - pars.vp_min) + pars.vp_min
                vint = vint_pred * 0
                vrms = vint_pred * 0
                valid = vint_pred * 0
                for jj in range(vint.shape[0]):
                    cmp0 = get_first_cmp_pos(gen.F.rec_pos[0,:], gen.F.src_pos[0,:], ds * pars.dh)
                    ind0 = int(cmp0 / pars.dh)
                    vint[jj, :] = interval_velocity_time(vp[:, ind0+jj * ds], pars=pars)[
                                 ::pars.resampling]
                    vrms[jj, :] = calculate_vrms(vp[:, ind0 + jj* ds], pars.dh,
                                                 pars.Npad, pars.NT, pars.dt, pars.tdelay,
                                                 pars.source_depth)[::pars.resampling]
                    z0 = int(pars.source_depth/pars.dh)
                    vid = int((2*np.sum(pars.dh/vp[z0:, ind0+jj * ds]) + pars.tdelay) /pars.dt /pars.resampling)
                    valid[jj, :vid] = 1

                ng = int(gen.F.rec_pos[0,:].shape[0] / gen.F.src_pos[0,:].shape[0])
                offsets = np.abs(gen.F.rec_pos[0,:ng] - gen.F.src_pos[0, 0])
                t = np.arange(0, data_cmp.shape[0]) * pars.dt * pars.resampling
                stack = np.zeros_like(vint)
                vrms_pred_smooth = medfilt(vrms_pred, [11, 1])

                savefile = h5.File(preddir + "/" + savefile + "_pred", "w")
                savefile['vint_pred'] = vint_pred
                savefile['vrms_pred'] = vrms_pred
                savefile['ref_pred'] = refpred
                savefile['vint'] = vint
                savefile['vrms'] = vrms
                savefile['valid'] = valid
                savefile['stack'] = stack
                savefile.close()

    rmse_vrms = np.zeros(ndatasets)
    rmse_vint = np.zeros(ndatasets)

    """
    __________________Take the mean of predictions of the ensemble______________
    """
    for ii in range(ndatasets):
        savefile = "example_%d" % ii
        vint_pred = 0
        vrms_pred = 0
        n = 0
        for logdir in logdirs:
            preddir = os.path.join(savedir, logdir)
            predictions = fnmatch.filter(os.listdir(preddir), 'example_*_pred')
        
            if savefile in examples and (savefile + "_pred") in predictions:
                savefile = h5.File(preddir + "/" + savefile + "_pred", "r")
                vint_pred += np.transpose(savefile['vint_pred'][:])
                vrms_pred += np.transpose(savefile['vrms_pred'][:])
                vint = np.transpose(savefile['vint'][:])
                vrms = np.transpose(savefile['vrms'][:])
                valid = np.transpose(savefile['valid'][:])
                savefile.close()
                n += 1
        for jj in range(vint.shape[1]):
            ind0 = np.nonzero(vint - vint[0,jj])[0][0]
            valid[0:ind0, jj] = 0

        vint_pred = vint_pred / n
        vrms_pred = vrms_pred / n
        rmse_vint[ii] = np.sqrt(np.sum(valid*((vint_pred - vint))**2)/np.sum(valid))
        rmse_vrms[ii] = np.sqrt(np.sum(valid * ((vrms_pred - vrms)) ** 2) / np.sum(valid))

    sort_rmses = np.argsort(rmse_vint)
    perc10 = sort_rmses[int(len(sort_rmses) * 0.1)]
    perc50 = sort_rmses[int(len(sort_rmses) * 0.5)]
    perc90 = sort_rmses[int(len(sort_rmses) * 0.8)]
    percs = [perc10, perc50, perc90]

    NX = vint_pred.shape[1]
    NZ = vint_pred.shape[0]
    ds = 50

    """
    _____________________________Create the plot_______________________________
    """
    def plot_model(thisax, v, label, with_ylabel=True, tmin=0, tmax=8, noyaxis=False):
        im = thisax.imshow(v / 1000, cmap=plt.get_cmap("jet"),
                           aspect="auto", interpolation="bilinear",
                           vmin=pars.vp_min / 1000, vmax=pars.vp_max / 1000,
                           extent=[0, (NX + 1) * ds / 1000,
                                   (NZ + 1) * pars.dt * pars.resampling, 0])
        thisax.set_xlabel('x (km)')
        if with_ylabel:
            thisax.set_ylabel('T (s)')
        thisax.set_ylim(top=tmin)
        thisax.set_ylim(bottom=tmax)
        thisax.yaxis.set_ticks(np.arange(tmin,tmax,2))
        if noyaxis:
            thisax.yaxis.set_ticks([])
        ymin, ymax =thisax.get_ylim()
        xmin, xmax = thisax.get_xlim()
        thisax.set_title(label, fontsize="medium")
        return im

    fig = plt.figure(figsize=(16/2.54, 8/2.54))
    gs = gridspec.GridSpec(nrows=5, ncols=55, height_ratios=[0.1, 1.2, 1.2, 10, 0.1])
    
    labels0 = [ "a)", "b)", "c)"]
    labels = [ "True", "Pred", "True", "Pred", "True", "Pred"]

    for ii, perc in enumerate(percs):
        savefile = "example_%d" % perc
        vint_pred = 0
        vrms_pred = 0
        n = 0
        for logdir in logdirs:
            preddir = os.path.join(savedir, logdir)
            if not os.path.isdir(preddir):
                os.makedirs(preddir)
                predictions = fnmatch.filter(os.listdir(preddir), 'example_*_pred')
                
            if savefile in examples and (savefile + "_pred") in predictions:
                savefile = h5.File(preddir + "/" + savefile + "_pred", "r")
                vint_pred += np.transpose(savefile['vint_pred'][:])
                vint = np.transpose(savefile['vint'][:])
                valid = np.transpose(savefile['valid'][:])
                savefile.close()
                n += 1
        vint_pred = vint_pred / n
        vint_pred = gaussian_filter(vint_pred, [3, 3])
        vint_pred[valid<1] = np.NaN
        
        if ii==0:
            with_ylabel=True
            noyaxis = False
        else:
             with_ylabel=False
             noyaxis = True
        ax = fig.add_subplot(gs[3, (19*ii):(19*ii+8)])
        plot_model(ax, vint, label=labels[2*ii], with_ylabel=with_ylabel, tmin=3, tmax=8.01, noyaxis=noyaxis)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.text(xmin - 0.2 * (xmax-xmin), ymax + 0.11*(ymax-ymin),
                       labels0[ii], fontsize="large")
        im = plot_model(fig.add_subplot(gs[3, (19*ii+9):(19*ii+17)]), vint_pred, label=labels[2*ii+1], with_ylabel=False, tmin=3, tmax=8.01, noyaxis=True)

    cax=fig.add_subplot(gs[1, 40:55])
    clr = plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    clr.set_ticks(np.arange(1.5, 4.1, 1.25))
    cax.xaxis.tick_top()
    cax.set_xlabel('V (km/s)', labelpad=10)
    cax.xaxis.set_label_position('top')

    savefile = "Paper/Fig/Case_article_predict2d"
    plt.savefig(savefile, dpi=600)
    plt.savefig(savefile+"_lowres", dpi=100)
    plt.show()
    
    
    print("Vint RMSE is %f m/s" % np.sqrt(np.mean(rmse_vint[rmse_vint!=9999]**2)))
    print("Vrms RMSE is %f m/s" % np.sqrt(np.mean(rmse_vrms[rmse_vint != 9999] ** 2)))





