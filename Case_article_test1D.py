#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Create the test dataset for Case article, performs the testing and plot results
"""
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator, generate_dataset
from vrmslearn.ModelGenerator import interval_velocity_time
from vrmslearn.Tester import Tester
from vrmslearn.RCNN import RCNN
from Cases_define import Case_article
import os
import argparse
import numpy as np
from plot_prediction import plot_predictions_semb3
import h5py as h5
import fnmatch
from scipy.stats import mode
import sys


def get_rms(name, masks, vint_pred, vint, vrms_pred, vrms, ref_pred, ref):
    print(name)
    masks = np.array(masks)
    nsamples = np.sum(masks == 1)
    vint_pred = np.array(vint_pred)
    vint = np.array(vint)
    vint_rmse = np.sqrt(np.sum(masks * (vint - vint_pred)**2) / nsamples)
    print("Interval velocity RMSE: %f m/s" % vint_rmse)
    
    vrms_pred = np.array(vrms_pred)
    vrms = np.array(vrms)
    vrms_rmse = np.sqrt(np.sum(masks * (vrms - vrms_pred) ** 2) / nsamples)
    print("RMS velocity RMSE: %f m/s" % vrms_rmse)
    
    ref_pred = np.array(ref_pred)
    ref = np.array(ref)
    nsamples = ref.flatten().shape[0]
    true_pos = np.sum(((ref - ref_pred) == 0) * (ref == 1)) / nsamples
    true_neg = np.sum(((ref - ref_pred) == 0) * (ref == 0)) / nsamples
    false_pos = np.sum((ref - ref_pred) == -1) / nsamples
    false_neg = np.sum((ref - ref_pred) == 1) / nsamples
    
    print("True positive: %f, True negative: %f, False positive %f "
          "False negative: %f" % (true_pos, true_neg, false_pos, false_neg))
        
    print("")

    return vint_rmse, vrms_rmse, true_pos, true_neg, false_pos, false_neg


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--nthread",
        type=int,
        default=1,
        help="Number of threads per gpus for data creation"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/model.ckpt-5000",
        help="Checkpoint filename for which to predict"
    )
    parser.add_argument(
        "--testing",
        type=int,
        default=3,
        help="1: testing only, 0: create dataset only, 2: testing+dataset, 3: ploting only"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="./seiscl_workdir",
        help="name of SeisCL working directory "
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset_article/test",
        help="name of the test dataset directory "
    )
    parser.add_argument(
            "--niter",
            type=int,
            default=1000,
            help="Iteration number of the checkpoint file"
            )


    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()
    
    dirs = []
    for dir1 in os.listdir('./'):
        if os.path.isdir(dir1):
            for dir2 in os.listdir(dir1):
                path2 = os.path.join(dir1, dir2)
                if os.path.isdir(path2):
                    dirs.append(path2)

    logdirs = fnmatch.filter(dirs, args.logdir)
    logdirs.sort()
    logdirs = [d + "/model.ckpt-" + str(args.niter) for d in logdirs]
    print("Found %d log directories to test" % len(logdirs), flush=True)
    for logdir in logdirs:
        print(logdir, flush=True)

    """
        __________________Define the parameters for Case Article________________
    """
    pars = Case_article()
   
    """
        __________________Generate the dataset______________________
    """
    pars.num_layers = 0
    dhmins = [5]
    layer_num_mins = [5, 10, 30, 50]
    nexamples = 400

    if not os.path.isdir(args.dataset_path):
        os.mkdir(args.dataset_path)

    n = 1
    if args.testing != 1 and args.testing != 3:
        for dhmin in dhmins:
           for layer_num_min in layer_num_mins:
               this_savepath = (args.dataset_path
                                + "/dhmin" + str(dhmin)
                                + "layer_num_min" + str(layer_num_min))
               pars.layer_dh_min = dhmin
               pars.layer_num_min = layer_num_min
               generate_dataset(pars=pars,
                                savepath=this_savepath,
                                nthread=1,
                                nexamples=nexamples,
                                workdir=args.workdir,
                                seed=n)
               n+=1

    if args.testing==0:
        sys.exit()

    """
        ___________________________Do the testing ______________________________
    """
    seismic_gen = SeismicGenerator(model_parameters=pars)
    nn = RCNN(input_size=seismic_gen.image_size,
              batch_size=2)
    tester = Tester(NN=nn, data_generator=seismic_gen)
    toeval = [nn.output_ref, nn.output_vrms, nn.output_vint]
    toeval_names = ["ref", "vrms", "vint"]
    vint_rmse_all = 0
    vrms_rmse_all = 0
    true_pos_all = 0
    true_neg_all = 0
    false_pos_all = 0
    false_neg_all = 0

    for dhmin in dhmins:
        for layer_num_min in layer_num_mins:
            vint = [None for _ in range(len(logdirs))]
            vint_pred = [None for _ in range(len(logdirs))]
            vrms = [None for _ in range(len(logdirs))]
            vrms_pred = [None for _ in range(len(logdirs))]
            ref = [None for _ in range(len(logdirs))]
            ref_pred = [None for _ in range(len(logdirs))]
            
            for n, logdir in enumerate(logdirs):
                seismic_gen.pars.layer_dh_min = dhmin
                seismic_gen.pars.layer_num_min = layer_num_min
                this_savepath = os.path.join(args.dataset_path, logdir) + "/dhmin" + str(dhmin) + "layer_num_min" + str(layer_num_min)
                dataset_path = args.dataset_path + "/dhmin" + str(dhmin) + "layer_num_min" + str(layer_num_min)
                if not os.path.isdir(this_savepath):
                    os.makedirs(this_savepath)
                
                if args.testing != 3:
                    tester.test_dataset(savepath=this_savepath,
                                        toeval=toeval,
                                        toeval_names=toeval_names,
                                        restore_from=logdir,
                                        testpath = dataset_path)
                vp, vint_pred[n], masks, lfiles, pfiles = tester.get_preds(labelname="vp",
                                                        predname="vint",
                                                        maskname="valid",
                                                        savepath=this_savepath,
                                                        testpath=dataset_path)
                vrms[n], vrms_pred[n], _, _ , _ = tester.get_preds(labelname="vrms",
                                                    predname="vrms",
                                                    savepath=this_savepath,
                                                    testpath=dataset_path)
                ref[n], ref_pred[n], _, _ , _ = tester.get_preds(labelname="tlabels",
                                                    predname="ref",
                                                    savepath=this_savepath,
                                                    testpath=dataset_path)
                vint[n] = [None for _ in range(len(vp))]
                for ii in range(len(vint[n])):
                    vint[n][ii] = interval_velocity_time(vp[ii], pars=pars)
                    vint[n][ii] = vint[n][ii][::pars.resampling]
                    vint_pred[n][ii] = vint_pred[n][ii]*(pars.vp_max - pars.vp_min) + pars.vp_min
                    vrms_pred[n][ii] = vrms_pred[n][ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
                    vrms[n][ii] = vrms[n][ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
                    ref_pred[n][ii] = np.argmax(ref_pred[n][ii], axis=1)
                    ind0 = np.nonzero(ref[n][ii])[0][0]
                    masks[ii][0:ind0] = 0
                vint[n] = np.array(vint[n])
                vint_pred[n] = np.array(vint_pred[n])
                vrms[n] = np.array(vrms[n])
                vrms_pred[n] = np.array(vrms_pred[n])
                ref[n] = np.array(ref[n])
                ref_pred[n] = np.array(ref_pred[n])
                
                name = "Results for dhmin= %f, layer_num_min= %f, NN %d" % (dhmin, layer_num_min, n)
#                get_rms(name, masks, vint_pred[n], vint[n], vrms_pred[n],
#                        vrms[n], ref_pred[n], ref[n])

            
            vint = np.mean(vint, axis=0)
            vint_pred_std = np.std(vint_pred, axis=0)
            vint_pred = np.mean(vint_pred, axis=0)
            vrms = np.mean(vrms, axis=0)
            vrms_pred_std = np.std(vrms_pred, axis=0)
            vrms_pred = np.mean(vrms_pred, axis=0)
            ref_pred = mode(ref_pred, axis=0).mode[0]
            ref = mode(ref, axis=0).mode[0]
            
            name = "Results for dhmin= %f, layer_num_min= %f, total" % (dhmin, layer_num_min)
            (vint_rmse, vrms_rmse, true_pos,
             true_neg, false_pos, false_neg) = get_rms(name, masks,
                                                       vint_pred, vint,
                                                       vrms_pred, vrms,
                                                       ref_pred, ref)
            print("Standard deviation for vint %f, m/s" % np.mean(vint_pred_std))
            print("Standard deviation for vrms %f, m/s" % np.mean(vrms_pred_std))
            vint_rmse_all += vint_rmse
            vrms_rmse_all += vrms_rmse
            true_pos_all += true_pos
            true_neg_all += true_neg
            false_pos_all += false_pos
            false_neg_all += false_neg
            
            masks = np.array(masks)
            rmses = np.sqrt(np.sum(masks * (vrms - vrms_pred) ** 2, axis=1) / np.sum(
                masks == 1, axis=1))
            sort_rmses = np.argsort(rmses)
            perc10 = sort_rmses[int(len(sort_rmses) * 0.1)]
            perc50 = sort_rmses[int(len(sort_rmses) * 0.5)]
            perc90 = sort_rmses[int(len(sort_rmses) * 0.9)]
            file = h5.File(lfiles[perc10], "r")
            data10 = file['data'][:]
            file.close()
            file = h5.File(lfiles[perc50], "r")
            data50 = file['data'][:]
            file.close()
            file = h5.File(lfiles[perc90], "r")
            data90 = file['data'][:]
            file.close()
            
            t10 = (np.nonzero(ref[perc10, :])[0][0] - 100) * pars.dt * pars.resampling
            t50 = (np.nonzero(ref[perc50, :])[0][0] - 100) * pars.dt * pars.resampling
            t90 = (np.nonzero(ref[perc90, :])[0][0] - 100) * pars.dt * pars.resampling

            plot_predictions_semb3([data10[:, :], data50[:, :], data90[:,:]],
                                [vrms[perc10, :], vrms[perc50, :], vrms[perc90, :]],
                                [vrms_pred[perc10, :], vrms_pred[perc50, :], vrms_pred[perc90, :]],
                                [ref[perc10, :], ref[perc50, :], ref[perc90, :]],
                                [ref_pred[perc10, :], ref_pred[perc50, :], ref_pred[perc90, :]],
                                [vint[perc10, :], vint[perc50, :], vint[perc90, :]],
                                [vint_pred[perc10, :], vint_pred[perc50, :], vint_pred[perc90, :]],
                                [masks[perc10, :], masks[perc50, :], masks[perc90, :]],
                                pars, clip=0.02, clipsemb=0.6, plot_semb=True,
                                vint_pred_std = [vint_pred_std[perc10, :], vint_pred_std[perc50, :], vint_pred_std[perc90, :]],
                                vpred_std = [vrms_pred_std[perc10, :], vrms_pred_std[perc50, :], vrms_pred_std[perc90, :]],
                                tmin = [t10, t50, t90],
                                textlabels=["$P_{10}$",
                                            "$P_{50}$",
                                            "$P_{90}$"],
                                savefile="Paper/Fig/Case_article_test_dhmin"+str(dhmin)+"_lnummin" +str(layer_num_min))

    n = len(dhmins) * len(layer_num_mins)
    print("Total Results")
    print("Interval velocity RMSE: %f m/s" % (vint_rmse_all/n))
    print("RMS velocity RMSE: %f m/s" % (vrms_rmse_all / n))
    print("True positive: %f, True negative: %f, False positive %f "
          "False negative: %f" % (true_pos_all/n,
                                  true_neg_all/n,
                                  false_pos_all/n,
                                  false_neg_all/n))

