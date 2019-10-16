#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Script to perform data creation and training for the main case presented in
    the article. For a smaller training set, see Case_small.py
"""

from vrmslearn.ModelParameters import ModelParameters
from Cases_define import Case_article
from vrmslearn.SeismicGenerator import SeismicGenerator, generate_dataset
from vrmslearn.Trainer import Trainer
from vrmslearn.RCNN import RCNN
import os
import argparse
import tensorflow as tf
import fnmatch

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--nthread",
        type=int,
        default=1,
        help="Number of threads for data creation"
    )
    parser.add_argument(
        "--nthread_read",
        type=int,
        default=1,
        help="Number of threads used as input producer"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory in which to store the checkpoints"
    )
    parser.add_argument(
        "--training",
        type=int,
        default=1,
        help="1: training only, 0: create dataset only, 2: training+dataset"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="./seiscl_workdir",
        help="name of SeisCL working directory "
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0008,
        help="learning rate "
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="epsilon for adadelta"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=40,
        help="size of the batches"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 for adadelta"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="beta2 for adadelta"
    )
    parser.add_argument(
        "--nmodel",
        type=int,
        default=1,
        help="Number of models to train"
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=1,
        help="1: Add noise to the data"
        )
    parser.add_argument(
        "--use_peepholes",
        type=int,
        default=1,
        help="1: Use peephole version of LSTM"
        )


    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    savepath = "./dataset_article"
    logdir = args.logdir
    nthread = args.nthread
    batch_size = args.batchsize

    """
        _______________________Define the parameters ______________________
    """
    pars = Case_article(noise=args.noise)

    """
        _______________________Generate the dataset_____________________________
    """
    gen = SeismicGenerator(model_parameters=pars)

    pars.num_layers = 0
    dhmins = [5]
    layer_num_mins = [5, 10, 30, 50]
    nexamples = 10000

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    if args.training != 1:
        for dhmin in dhmins:
            for layer_num_min in layer_num_mins:
                pars.layer_dh_min = dhmin
                pars.layer_num_min = layer_num_min
                this_savepath = (savepath
                                 + "/dhmin%d" % dhmin
                                 + "_layer_num_min%d" % layer_num_min)
                generate_dataset(pars=pars,
                                 savepath=this_savepath,
                                 nthread=args.nthread,
                                 nexamples=nexamples,
                                 workdir=args.workdir)


    """
        ___________________________Do the training _____________________________

     We define 3 stages for inversion, with different alpha, beta gamma in the
     loss function:
        1st stage:  alpha = 0, beta=1 and gamma=0: we train for reflection 
                    identification
        2nd stage:  alpha = 0.2, beta=0.1 and gamma=0.1: we train for reflection 
                    identification and vrms, with regularization on vrms time 
                    derivative (alpha) et higher weights on vrms at reflections
                    arrival times (gamma)
        3rd stage:  alpha = 0.02, beta=0.02 and gamma=0.1, we add weight to vrms

    """
    schedules = [[0.0, 0.95, 0, 0, 0],
                 [0.05, 0.1, 0, 0, 0],
                 [0.05, 0.1, 0, 0.35, 0.05],
                 [0.01, 0.01, 0, 0, 0],
                 [0, 0, 0, 0.95, 0.05]]
    niters = [1000, 10000, 10000, 1000, 1000]
    if args.training != 0:
        for nmod in range(args.nmodel):
            restore_from = None
            npass = 0
            for ii, schedule in enumerate(schedules):
                this_savepath = []
                for layer_num_min in layer_num_mins:
                    for dhmin in dhmins:
                        this_savepath.append(savepath
                                             + "/dhmin%d" % dhmin
                                             + "_layer_num_min%d" % layer_num_min)
                this_logdir = (logdir
                               + "%d" % nmod
                               + "/%d" % npass
                               + "_schedule%d" % ii
                               + "_lr%f_eps_%f" % (args.lr, args.eps)
                               + "_beta1%f" % args.beta1
                               + "_beta2%f" % args.beta2
                               + "_batch_size_%d" % batch_size)

                lastfile = this_logdir + 'model.ckpt-' + str(niters[ii]) + '*'

                try:
                    isckpt = fnmatch.filter(os.listdir(this_logdir),
                                  'model.ckpt-' + str(niters[ii]) + '*')
                except FileNotFoundError:
                    isckpt =[]

                if not isckpt:
                    print(this_logdir)
                    pars.layer_dh_min = dhmin
                    pars.layer_num_min = layer_num_min
                    seismic_gen = SeismicGenerator(model_parameters=pars)
                    nn = RCNN(input_size=seismic_gen.image_size,
                              batch_size=batch_size,
                              alpha=schedule[0],
                              beta=schedule[1],
                              gamma=schedule[2],
                              zeta=schedule[3],
                              omega=schedule[4],
                              use_peepholes=args.use_peepholes)

                    if layer_num_min == layer_num_mins[0] and dhmin == dhmins[0]:
                        learning_rate = args.lr
                    else:
                        learning_rate = args.lr/8
                    if ii>2:
                        learning_rate = args.lr/128
                    # Optimize only last layers during last schedule
                    if ii == 4:
                        with nn.graph.as_default():
                            var_to_minimize = tf.trainable_variables(scope='rnn_vint')
                            var_to_minimize.append(tf.trainable_variables(scope='Decode_vint'))
                    else:
                        var_to_minimize = None
                
                    trainer = Trainer(NN=nn,
                                      data_generator=seismic_gen,
                                      checkpoint_dir=this_logdir,
                                      learning_rate=learning_rate,
                                      beta1=args.beta1,
                                      beta2=args.beta2,
                                      epsilon=args.eps,
                                      var_to_minimize=var_to_minimize)
                    trainer.train_model(niter=niters[ii],
                                        savepath=this_savepath,
                                        restore_from=restore_from,
                                        thread_read=args.nthread_read)
                restore_from = this_logdir + '/model.ckpt-' + str(niters[ii])
                npass += 1
