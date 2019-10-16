#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot the loss as a function of epoch (Reproduces Figure 2 of the article)
"""
import tensorflow as tf
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
import fnmatch
import os
import numpy as np
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator
from vrmslearn.Tester import Tester
from vrmslearn.RCNN import RCNN
from vrmslearn.ModelGenerator import interval_velocity_time
import h5py as h5

def get_test_error(dirlog, savepath, dataset_path):
    """
        Compute the error on a test set
        
        @params:
        dirlog (str) : Directory containing the trained model.
        savepath (str): Directory in which to save the predictions
        dataset_path (str): Directory of the test set
        
        @returns:
        vrms_rmse (float) : RMSE for Vrms
        vint_rmse: RMSE for Vint
        true_pos: Primary reflection identification: Ratio of true positive
        true_neg: Primary reflection identification: Ratio of true negative
        false_pos: Primary reflection identification: Ratio of false positive
        false_neg: Primary reflection identification: Ratio of false negative
    """
    
    if os.path.isfile(savepath + ".hdf5"):
        file = h5.File(savepath + ".hdf5")
        vint_rmse = file["vint_rmse"].value
        vrms_rmse = file["vrms_rmse"].value
        true_pos = file["true_pos"].value
        true_neg = file["true_neg"].value
        false_pos = file["false_pos"].value
        false_neg = file["false_neg"].value
        file.close()
    else:
        pars = ModelParameters()
        pars.read_parameters_from_disk(dataset_path+"/model_parameters.hdf5")
        seismic_gen = SeismicGenerator(model_parameters=pars)
        nn = RCNN(input_size=seismic_gen.image_size,
                  batch_size=100)
        tester = Tester(NN=nn, data_generator=seismic_gen)
        toeval = [nn.output_ref, nn.output_vrms, nn.output_vint]
        toeval_names = ["ref", "vrms", "vint"]
        vint_rmse_all = 0
        vrms_rmse_all = 0
        true_pos_all = 0
        true_neg_all = 0
        false_pos_all = 0
        false_neg_all = 0
        
        tester.test_dataset(savepath=savepath,
                          toeval=toeval,
                          toeval_names=toeval_names,
                          restore_from=dirlog,
                          testpath = dataset_path)
        vp, vint_pred, masks, lfiles, pfiles = tester.get_preds(labelname="vp",
                                                              predname="vint",
                                                              maskname="valid",
                                                              savepath=savepath,
                                                              testpath = dataset_path)
        vrms, vrms_pred, _, _ , _ = tester.get_preds(labelname="vrms",
                                                   predname="vrms",
                                                   savepath=savepath,
                                                     testpath = dataset_path)
        ref, ref_pred, _, _ , _ = tester.get_preds(labelname="tlabels",
                                                 predname="ref",
                                                 savepath=savepath,
                                                   testpath = dataset_path)
        vint = [None for _ in range(len(vp))]
        for ii in range(len(vint)):
            vint[ii] = interval_velocity_time(vp[ii], pars=pars)
            vint[ii] = vint[ii][::pars.resampling]
            vint_pred[ii] = vint_pred[ii]*(pars.vp_max - pars.vp_min) + pars.vp_min
            vrms_pred[ii] = vrms_pred[ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
            vrms[ii] = vrms[ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
            ref_pred[ii] = np.argmax(ref_pred[ii], axis=1)
            ind0 = np.nonzero(ref[ii])[0][0]
            masks[ii][0:ind0] = 0
        vint = np.array(vint)
        vint_pred = np.array(vint_pred)
        vrms = np.array(vrms)
        vrms_pred = np.array(vrms_pred)
        ref = np.array(ref)
        ref_pred = np.array(ref_pred)



        masks = np.array(masks)
        nsamples = np.sum(masks == 1)
        vint_rmse = np.sqrt(np.sum(masks * (vint - vint_pred)**2) / nsamples)
        vrms_rmse = np.sqrt(np.sum(masks * (vrms - vrms_pred) ** 2) / nsamples)

        nsamples = ref.flatten().shape[0]
        true_pos = np.sum(((ref - ref_pred) == 0) * (ref == 1)) / nsamples
        true_neg = np.sum(((ref - ref_pred) == 0) * (ref == 0)) / nsamples
        false_pos = np.sum((ref - ref_pred) == -1) / nsamples
        false_neg = np.sum((ref - ref_pred) == 1) / nsamples

        file = h5.File(savepath + ".hdf5")
        file["vint_rmse"] = vint_rmse
        file["vrms_rmse"] = vrms_rmse
        file["true_pos"] = true_pos
        file["true_neg"] = true_neg
        file["false_pos"] = false_pos
        file["false_neg"] = false_neg
        file.close()

    return vrms_rmse, vint_rmse, true_pos, true_neg, false_pos, false_neg

if __name__ == "__main__":


    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--logdir",
        type=str,
        default="Case_article0",
        help="name of the directory to save logs : str"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset_article/test/dhmin5layer_num_min10",
        help="path of the test dataset"
        )
   
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()
    training_size = 40000
    batch_size = 40
    savefile = "Paper/Fig/Case4_loss"
    
    
    # Obtain all subdirectories containing tensorflow models inside args.logdir.
    dirs = []
    dir_models = {}
    for dir1 in os.listdir(args.logdir):
        path1 = os.path.join(args.logdir, dir1)
        if os.path.isdir(path1):
            files = []
            for dir2 in os.listdir(path1):
                path2 = os.path.join(path1, dir2)
                if os.path.isfile(path2):
                    files.append(path2)
            efiles = fnmatch.filter(files, os.path.join(path1,"events.*"))
            efiles.sort()
            dirs.append(efiles)
            allmodels = fnmatch.filter(files, os.path.join(path1,"model.ckpt-*.meta"))
            allmodels.sort()
            dir_models[dirs[-1][-1]] = [a[:-5] for a in allmodels]
    for dir in dirs:
        print(dir)

    # Create the figure
    fig, ax = plt.subplots(3, 1, figsize=[8 / 2.54, 12 / 2.54])
    step0= 0
    plots = [[] for _ in range(3)]
    labels = ["Phase 0", "Phase 1", "Phase 2"]
    for ii, dir in enumerate(dirs[:-2]):
        step = []
        loss = []
        # Get Loss for each stage of training and each iteration
        for e in dir:
            for summary in tf.train.summary_iterator(e):
                for v in summary.summary.value:
                    if v.tag == 'Loss_Function/loss':
                        loss.append(v.simple_value)
                        step.append(summary.step + step0)
        inds = np.argsort(step)
        step = np.array(step)[inds][1:]
        loss = np.array(loss)[inds][1:]
        plots[ii], = ax[0].semilogy(step * batch_size /training_size, loss, basey=2)

        if ii!=0:
            steprms0 = steprms[-1]
            vrms0 = vrms[-1]
            vint0 = vint[-1]
        
        # Compute test set error for each model during training (or retrieve it)
        steprms = []
        vrms = []
        vint = []
        for dirlog in dir_models[dir[-1]]:
            savepath = dirlog + "_test/" + args.dataset_path
            if not os.path.isdir(savepath):
                os.makedirs(savepath)

            vrms_rmse, vint_rmse, _, _, _, _ = get_test_error(dirlog, savepath, args.dataset_path)
            steprms.append(int(dirlog.split("-")[-1]) + step0)
            vrms.append(vrms_rmse)
            vint.append(vint_rmse)
        inds = np.argsort(steprms)
        steprms = np.array(steprms)[inds][1:]
        vrms = np.array(vrms)[inds][1:]
        vint = np.array(vint)[inds][1:]
        if ii!=0:
            steprms = np.insert(steprms, 0, steprms0)
            vrms = np.insert(vrms, 0, vrms0)
            vint = np.insert(vint, 0, vint0)
        ax[1].plot(steprms * batch_size /training_size, vrms)
        ax[2].plot(steprms * batch_size /training_size, vint)
    
        step0 = step[-1]

    # Figure presentation
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("RMSE (m/s)")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("RMSE (m/s)")
    ax[0].legend(plots, labels,
                 loc='upper right',
                 bbox_to_anchor=(1.15, 1.35),
                 handlelength=0.4)
    plt.tight_layout(rect=[0.001, 0, 0.9999, 1])
    plt.savefig(savefile, dpi=600)
    plt.savefig(savefile+"_lowres", dpi=100)
    plt.show()


