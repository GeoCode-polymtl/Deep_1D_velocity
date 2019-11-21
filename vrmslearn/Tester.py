#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class tests a NN on a dataset.
"""
from vrmslearn.RCNN import RCNN
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.Inputqueue import BatchManager
from vrmslearn.SeismicGenerator import SeismicGenerator
import tensorflow as tf
import time
import os
import fnmatch
import h5py as h5
import copy

class Tester(object):
    """
    This class tests a NN on a dataset.
    """

    def __init__(self,
                 NN: RCNN,
                 data_generator: SeismicGenerator):
        """
        Initialize the tester

        @params:
        NN (RCNN) : A tensforlow neural net
        data_generator (SeismicGenerator): A data generator object

        @returns:
        """
        self.NN = NN
        self.data_generator = data_generator

    def test_dataset(self,
                     savepath: str,
                     toeval: list,
                     toeval_names: list,
                     testpath: str = None,
                     filename: str = 'example_*',
                     restore_from: str = None,
                     tester_n: int = 0):
        """
        This method evaluate predictions on all examples contained in savepath,
        and save the predictions in hdf5 files.

        @params:
        savepath (str) : The path in which the test examples are found
        filename (str): The structure of the examples' filenames
        toeval (list): List of tensors to predict
        restore_from (str): File containing the trained weights
        tester_n (int): Label for the model to use for prediction

        @returns:
        """
        
        if testpath is None:
            testpath = savepath
        # Do the testing
        examples = fnmatch.filter(os.listdir(testpath), filename)
        predictions = fnmatch.filter(os.listdir(savepath), filename)
        with self.NN.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, restore_from)
                batch = []
                bexamples = []
                for ii, example in enumerate(examples):
                    predname = example + "_pred" + str(tester_n)
                    if "pred" not in example and predname not in predictions:
                        bexamples.append(example)
                        batch.append(self.data_generator.read_example(savedir=testpath,
                                                                      filename=example))

                    if len(batch) == self.NN.batch_size:
                        batch = self.data_generator.aggregate_examples(batch)
                        feed_dict = dict(zip(self.NN.feed_dict, batch))
                        evaluated = sess.run(toeval, feed_dict=feed_dict)
                        for jj, bexample in enumerate(bexamples):
                            savefile = h5.File(savepath + "/" + bexample + "_pred" + str(tester_n), "w")
                            for kk, el in enumerate(toeval_names):
                                savefile[el] = evaluated[kk][jj, :]
                            savefile.close()
                        batch = []
                        bexamples = []

    def get_preds(self,
                  labelname: str,
                  predname: str,
                  savepath: str,
                  testpath: str = None,
                  filename: str = 'example_*',
                  maskname: str = None,
                  tester_n: int = 0):
        """
        This method returns the labels and the predictions for an output.

        @params:
        labelname (str) : Name of the labels in the example file
        predname (str) : Name of the predictions in the example file
        maskname(str) : name of the valid predictions mask
        savepath (str) : The path in which the test examples are found
        filename (str): The structure of the examples' filenames
        tester_n (int): Label for the model to use for prediction

        @returns:
        labels (list): List containing all labels
        preds (list):  List containing all predictions
        """
        
        if testpath is None:
            testpath = savepath
        examples = fnmatch.filter(os.listdir(testpath), filename)
        predictions = fnmatch.filter(os.listdir(savepath), filename)
        labels = []
        preds = []
        masks = []
        lfiles = []
        pfiles = []
        for ii, example in enumerate(examples):
            example_pred = example + "_pred" + str(tester_n)
            if "pred" not in example and example_pred in predictions:
                pfiles.append(savepath + "/" + example_pred)
                pfile = h5.File(pfiles[-1], "r")
                preds.append(pfile[predname][:])
                pfile.close()
                lfiles.append(testpath + "/" + example)
                lfile = h5.File(lfiles[-1], "r")
                labels.append(lfile[labelname][:])
                if maskname is not None:
                    masks.append(lfile[maskname][:])
                lfile.close()



        return labels, preds, masks, lfiles, pfiles
