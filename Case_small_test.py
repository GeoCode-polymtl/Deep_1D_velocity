#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Create the test dataset for Case 1, performs the testing and plot results
"""

from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicGenerator import SeismicGenerator, generate_dataset
from vrmslearn.ModelGenerator import interval_velocity_time
from vrmslearn.Tester import Tester
from vrmslearn.RCNN import RCNN
from plot_prediction import plot_predictions_semb3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--nthread",
        type=int,
        default=3,
        help="Number of threads per gpus for data creation"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="logs/model.ckpt-5000",
        help="Checkpoint filename for which to predict"
    )
    parser.add_argument(
        "--testing",
        type=int,
        default=2,
        help="1: testing only, 0: create dataset only, 2: testing+dataset,  3: ploting only"
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
        default="./dataset_1/test",
        help="name of SeisCL working directory "
    )


    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()


    savepath = args.dataset_path

    """
        __________________Define the parameters for Case 1______________________
    """
    pars = ModelParameters()
    pars.num_layers = 0
    dhmins = [40, 30, 20]
    layer_num_mins = [5, 12]
    nexamples = 10

    """
        _______________________Generate the dataset_____________________________
    """
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    n = 1
    if args.testing != 1:
        for dhmin in dhmins:
            for layer_num_min in layer_num_mins:
                pars.layer_dh_min = dhmin
                pars.layer_num_min = layer_num_min
                this_savepath = savepath + "/dhmin" + str(dhmin) + "layer_num_min" + str(layer_num_min)

                generate_dataset(pars=pars,
                                 savepath=this_savepath,
                                 nthread=1,
                                 nexamples=nexamples,
                                 workdir=args.workdir,
                                 seed=n)
                n += 1

    """
        ___________________________Do the testing ______________________________
    """
    seismic_gen = SeismicGenerator(model_parameters=pars)
    nn = RCNN(input_size=seismic_gen.image_size,
              batch_size=2)
    tester = Tester(NN=nn, data_generator=seismic_gen)
    toeval = [nn.output_ref, nn.output_vrms, nn.output_vint]
    toeval_names = ["ref", "vrms", "vint"]
    vint_rmse = 0
    vrms_rmse = 0
    true_pos_all = 0
    true_neg_all = 0
    false_pos_all = 0
    false_neg_all = 0
    for dhmin in dhmins:
        for layer_num_min in layer_num_mins:
            this_savepath = savepath + "/dhmin" + str(dhmin) + "layer_num_min" + str(layer_num_min)
            if args.testing != 3:
                tester.test_dataset(savepath=this_savepath,
                                    toeval=toeval,
                                    toeval_names=toeval_names,
                                    restore_from=args.model_file)
            vp, vint_pred, masks, lfiles, pfiles = tester.get_preds(labelname="vp",
                                                    predname="vint",
                                                    maskname="valid",
                                                    savepath=this_savepath)
            vrms, vrms_pred, _, _, _ = tester.get_preds(labelname="vrms",
                                                        predname="vrms",
                                                        savepath=this_savepath)
            ref, ref_pred, _, _, _ = tester.get_preds(labelname="tlabels",
                                                      predname="ref",
                                                      savepath=this_savepath)

            vint = [None] * len(vp)
            for ii in range(len(vint)):
                vint[ii] = interval_velocity_time(vp[ii], pars=pars)
                vint[ii] = vint[ii][::pars.resampling]
                vint_pred[ii] = vint_pred[ii]*(pars.vp_max - pars.vp_min) + pars.vp_min
                vrms_pred[ii] = vrms_pred[ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
                vrms[ii] = vrms[ii] * (pars.vp_max - pars.vp_min) + pars.vp_min
                ref_pred[ii] = np.argmax(ref_pred[ii], axis=1)
                #plt.plot(vint[ii])
                #plt.plot(vint_pred[ii])
                #plt.show()


            print("Results for dhmin= %f, layer_num_min= %f" % (dhmin, layer_num_min))
            masks = np.array(masks)
            nsamples = np.sum(masks == 1)
            vint_pred = np.array(vint_pred)
            vint = np.array(vint)
            rmse = np.sqrt(np.sum(masks * (vint - vint_pred)**2) / nsamples)
            vint_rmse += rmse
            print("Interval velocity RMSE: %f m/s" % rmse)


            vrms_pred = np.array(vrms_pred)
            vrms = np.array(vrms)
            rmse = np.sqrt(np.sum(masks * (vrms - vrms_pred) ** 2) / nsamples)
            vrms_rmse += rmse
            print("RMS velocity RMSE: %f m/s" % rmse)

            ref_pred = np.array(ref_pred)
            ref = np.array(ref)
            nsamples = ref.flatten().shape[0]
            true_pos = np.sum(((ref - ref_pred) == 0) * (ref == 1)) / nsamples
            true_neg = np.sum(((ref - ref_pred) == 0) * (ref == 0)) / nsamples
            false_pos = np.sum((ref - ref_pred) == -1) / nsamples
            false_neg = np.sum((ref - ref_pred) == 1) / nsamples

            true_pos_all += true_pos
            true_neg_all += true_neg
            false_pos_all += false_pos
            false_neg_all += false_neg

            print("True positive: %f, True negative: %f, False positive %f "
                  "False negative: %f" % (true_pos, true_neg, false_pos, false_neg))

            print("")

            rmses = np.sqrt(np.sum(masks * (vint - vint_pred) ** 2, axis=1) / np.sum(
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

            plot_predictions_semb3([data10, data50, data90],
                                   [vrms[perc10, :], vrms[perc50, :], vrms[perc90, :]],
                                   [vrms_pred[perc10, :], vrms_pred[perc50, :], vrms_pred[perc90, :]],
                                   [ref[perc10, :], ref[perc50, :], ref[perc90, :]],
                                   [ref_pred[perc10, :], ref_pred[perc50, :], ref_pred[perc90, :]],
                                   [vint[perc10, :], vint[perc50, :], vint[perc90, :]],
                                   [vint_pred[perc10, :], vint_pred[perc50, :], vint_pred[perc90, :]],
                                   [masks[perc10, :], masks[perc50, :], masks[perc90, :]],
                                   pars,
                                   savefile="Paper/Fig/Case1_test_dhmin"+str(dhmin)+"_lnummin" +str(layer_num_min))


    n = len(dhmins) * len(layer_num_mins)
    print("Total Results")
    print("Interval velocity RMSE: %f m/s" % (vint_rmse/n))
    print("RMS velocity RMSE: %f m/s" % (vrms_rmse / n))
    print("True positive: %f, True negative: %f, False positive %f "
          "False negative: %f" % (true_pos_all/n,
                                  true_neg_all/n,
                                  false_pos_all/n,
                                  false_neg_all/n))