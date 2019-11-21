#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 1D prediction of RMS and interval velocity
in time.
"""
import tensorflow as tf
import numpy as np


class RCNN(object):
    """
    This class build a NN based on recursive CNN that can identify primarary
    reflections on a CMP gather and ouput the RMS and interval velocity in time
    """

    def __init__(self,
                 input_size: list=[0, 0],
                 batch_size: int=1,
                 alpha: float = 0,
                 beta: float = 0,
                 gamma: float = 0,
                 zeta: float = 0,
                 omega: float = 0,
                 dec: float = 0,
                 use_peepholes = False,
                 with_masking: bool = False):
        """
        Build the neural net in tensorflow, along the cost function

        @params:
        input_size (list): the size of the CMP [NT, NX]
        batch_size (int): Number of CMPs in a batch
        alpha (float): Fraction of the loss dedicated to vrms derivative in time
        beta (float): Fraction of the loss dedicated to primary identification
        gamma (float): Fraction of the loss dedicated vrms  at reflection times
        zeta (float): Fraction of the loss dedicated interval velocity
        omega (float): Fraction of the loss dedicated vint derivative in time
        with_masking (bool): If true, masks random part of the CMPs

        @returns:
        """

        self.input_size = input_size
        self.graph = tf.Graph()
        self.with_masking = with_masking
        self.feed_dict = []
        self.batch_size = batch_size
        self.use_peepholes = use_peepholes
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            (self.input,
             self.label_vrms,
             self.weights,
             self.label_ref,
             self.label_vint) = self.generate_io()
            self.feed_dict = [self.input,
                              self.label_vrms,
                              self.weights,
                              self.label_ref,
                              self.label_vint]
            self.input_scaled = self.scale_input()
            (self.output_vrms,
             self.output_ref,
             self.output_vint) = self.build_neural_net()
            self.loss = self.define_loss(alpha=alpha,
                                         beta=beta,
                                         gamma=gamma,
                                         zeta=zeta,
                                         omega=omega,
                                         dec=dec)


    def generate_io(self):
        """
        This method creates the input nodes.

        @params:

        @returns:
        input_data (tf.tensor)  : Placeholder of CMP gather.
        label_vrms (tf.placeholder) : Placeholder of RMS velocity labels.
        weights (tf.placeholder) : Placeholder of time weights
        label_ref (tf.placeholder) : Placeholder of primary reflection labels.
        label_vint (tf.placeholder) : Placeholder of interval velocity labels.
        """

        with tf.name_scope('Inputs'):
            # Create placeholder for input
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[self.batch_size,
                                               self.input_size[0],
                                               self.input_size[1],
                                               1],
                                        name='data')

            label_vrms = tf.placeholder(dtype=tf.float32,
                                    shape=[self.batch_size,
                                           self.input_size[0]],
                                    name='vrms')

            weights = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size,
                                            self.input_size[0]],
                                     name='weigths')

            label_ref = tf.placeholder(dtype=tf.int32,
                                     shape=[self.batch_size,
                                            self.input_size[0]],
                                     name='reflections')
            label_vint = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch_size,
                                                self.input_size[0]],
                                          name='vint')

        return input_data, label_vrms, weights, label_ref, label_vint

    def scale_input(self):
        """
        Scale each trace to its RMS value, and each CMP to is RMS

        @params:

        @returns:
        input_data (tf.tensor)  : Placeholder of CMP gather.
        label_vrms (tf.placeholder) : Placeholder of RMS velocity labels.
        weights (tf.placeholder) : Placeholder of time weights
        label_ref (tf.placeholder) : Placeholder of primary reflection labels.
        label_vint (tf.placeholder) : Placeholder of interval velocity labels.
        """
        scaled = self.input / (tf.sqrt(reduce_sum(self.input ** 2, axis=[1],
                                       keepdims=True))
                               + np.finfo(np.float32).eps)

#        scaled = scaled / tf.sqrt(reduce_sum(scaled ** 2,
#                                                axis=[1, 2],
#                                                keepdims=True))

        scaled = 1000*scaled / tf.reduce_max(scaled, axis=[1, 2], keepdims=True)

        return scaled

    def build_neural_net(self):
        """
        This method build the neural net in Tensorflow

        @params:

        @returns:
        decode_rms (tf.tensor) : RMS velocity predictions
        decode_ref (tf.tensor) : Primary reflections predictions
        decode_vint (tf.tensor) : Interval velocity predictions.
        """

        rnn_hidden = 200
        weights = [tf.Variable(tf.random_normal([15, 1, 1, 16], stddev=1e-1),
                               name='w1'),
                   tf.Variable(tf.random_normal([1, 9, 16, 16], stddev=1e-1),
                               name='w2'),
                   tf.Variable(tf.random_normal([15, 1, 16, 32], stddev=1e-1),
                               name='w3'),
                   tf.Variable(tf.random_normal([1, 9, 32, 32], stddev=1e-1),
                               name='w4'),
                   tf.Variable(tf.random_normal([15, 3, 32, 32], stddev=1e-2),
                               name='w5'),
                   tf.Variable(tf.random_normal([1, 2, 32, 32], stddev=1e-0),
                               name='w6')]

        biases = [tf.Variable(tf.zeros([16]), name='b1'),
                  tf.Variable(tf.zeros([16]), name='b2'),
                  tf.Variable(tf.zeros([32]), name='b3'),
                  tf.Variable(tf.zeros([32]), name='b4'),
                  tf.Variable(tf.zeros([32]), name='b5'),
                  tf.Variable(tf.zeros([32]), name='b6')]

        weightsr = [tf.Variable(tf.random_normal([15, 1, 1, 16], stddev=1e-1),
                               name='w1'),
                   tf.Variable(tf.random_normal([1, 9, 16, 16], stddev=1e-1),
                               name='w2'),
                   tf.Variable(tf.random_normal([15, 1, 16, 32], stddev=1e-1),
                               name='w3'),
                   tf.Variable(tf.random_normal([1, 9, 32, 32], stddev=1e-1),
                               name='w4'),
                   tf.Variable(tf.random_normal([15, 3, 32, 32], stddev=1e-2),
                               name='w5'),
                   tf.Variable(tf.random_normal([1, 2, 32, 32], stddev=1e-0),
                               name='w6')]

        biasesr = [tf.Variable(tf.zeros([16]), name='b1'),
                  tf.Variable(tf.zeros([16]), name='b2'),
                  tf.Variable(tf.zeros([32]), name='b3'),
                  tf.Variable(tf.zeros([32]), name='b4'),
                  tf.Variable(tf.zeros([32]), name='b5'),
                  tf.Variable(tf.zeros([32]), name='b6')]

        data_stream = self.input_scaled
        allout = [self.input_scaled]
        with tf.name_scope('Encoder'):
            for ii in range(len(weights) - 2):
                with tf.name_scope('CNN_' + str(ii)):
                    data_stream = tf.nn.relu(
                        tf.nn.conv2d(data_stream,
                                     weights[ii],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME') + biases[ii])
                    allout.append(data_stream)
        self.output_encoder = data_stream

        with tf.name_scope('Time_RCNN'):
            for ii in range(7):
                data_stream = tf.nn.relu(
                    tf.nn.conv2d(data_stream,
                                 weights[-2],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME') + biases[-2])
                allout.append(data_stream)

        self.output_time_rcnn = data_stream

        decode_stream = data_stream
        with tf.name_scope('Decoder'):
            n = -2
            for ii in range(7):
                decode_stream = tf.nn.conv2d_transpose(tf.nn.relu(decode_stream) + biasesr[-2],
                                                        weightsr[-2],
                                                        output_shape=allout[n].shape,
                                                        strides=[1, 1, 1, 1],
                                                        padding='SAME')
                                           
                n -= 1
            for ii in range(len(weights) - 2):
                decode_stream = tf.nn.conv2d_transpose(tf.nn.relu(decode_stream) + biasesr[len(weights) - 3 - ii],
                                                        weightsr[len(weights) - 3 - ii],
                                                        output_shape=allout[n].shape,
                                                        strides=[1, 1, 1, 1],
                                                        padding='SAME')
                                       
                n -= 1
        self.decoded = decode_stream

        with tf.name_scope('Offset_RCNN'):
            while data_stream.get_shape()[2] > 1:
                data_stream = tf.nn.relu(
                    tf.nn.conv2d(data_stream,
                                 weights[-1],
                                 strides=[1, 1, 2, 1],
                                 padding='VALID') + biases[-1])
        data_stream = reduce_max(data_stream, axis=[2], keepdims=False)
        self.output_offset_rcnn = data_stream


        output_size = int(data_stream.get_shape()[-1])
        with tf.name_scope('Decode_refevent'):
            decode_refw = tf.Variable(
                initial_value=tf.random_normal([output_size, 2],
                                               stddev=1e-4),
                name='decode_ref')
            final_projection = lambda x: tf.matmul(x, decode_refw)
            decode_ref = tf.map_fn(final_projection, data_stream)
    
        with tf.name_scope('RNN_vrms'):
            cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden, state_is_tuple=True, use_peepholes=self.use_peepholes)
            state0 = cell.zero_state(data_stream.get_shape()[0], tf.float32)
            data_stream, rnn_states = tf.nn.dynamic_rnn(cell, data_stream,
                                                        initial_state=state0,
                                                        time_major=False,
                                                        scope="rnn_vrms")
            self.rnn_vrms_out = data_stream

        with tf.name_scope('Decode_rms'):
            output_size = int(data_stream.get_shape()[-1])
            decode_rmsw = tf.Variable(
                initial_value=tf.random_normal([output_size, 1], stddev=1e-4),
                                               name='decode_rms')
            final_projection = lambda x: tf.matmul(x, decode_rmsw)
            decode_rms = tf.map_fn(final_projection, data_stream)
            decode_rms = decode_rms[:, :, 0]


        with tf.name_scope('RNN_vint'):
                cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden, state_is_tuple=True, use_peepholes=self.use_peepholes)
                state0 = cell.zero_state(data_stream.get_shape()[0], tf.float32)
                data_stream, rnn_states = tf.nn.dynamic_rnn(cell, data_stream,
                                                            initial_state=state0,
                                                            time_major=False,
                                                            scope="rnn_vint")
                self.rnn_vint_out = data_stream

        with tf.name_scope('Decode_vint'):
            output_size = int(data_stream.get_shape()[-1])
            decode_vintw = tf.Variable(
                initial_value=tf.random_normal([output_size, 1], stddev=1e-4),
                                               name='decode_vint')
            final_projection = lambda x: tf.matmul(x, decode_vintw)
            decode_vint = tf.map_fn(final_projection, data_stream)
            decode_vint = decode_vint[:, :, 0]

        self.allout = allout
        return decode_rms, decode_ref, decode_vint

    def define_loss(self, alpha=0.2, beta=0.1, gamma=0, zeta=0, omega=0, dec=0):
        """
        This method creates a node to compute the loss function.
        The loss is normalized.

        @params:

        @returns:
        loss (tf.tensor) : Output of node calculating loss.
        """
        with tf.name_scope("Loss_Function"):

            losses = []

            fact1 = (1 - alpha - beta - gamma - zeta - omega)

            # Calculate mean squared error of continuous rms velocity
            if fact1 > 0:
                num = tf.reduce_sum(self.weights*(self.label_vrms
                                                  - self.output_vrms) ** 2)
                den = tf.reduce_sum(self.weights*self.label_vrms ** 2)
                losses.append(fact1 * num / den)

            #  Calculate mean squared error of the derivative of the continuous
            # rms velocity(normalized)
            if alpha > 0:
                dlabels = self.label_vrms[:, 1:] - self.label_vrms[:, :-1]
                dout = self.output_vrms[:, 1:] - self.output_vrms[:, :-1]
                num = tf.reduce_sum(self.weights[:,:-1]*(dlabels - dout) ** 2)
                den = tf.reduce_sum(self.weights[:,:-1]*dlabels ** 2 + 0.000001)
                losses.append(alpha * num / den)

            #  Logistic regression of zero offset time arrival of reflections
            if beta > 0:
                if self.with_masking:
                    weightsr = tf.expand_dims(self.weights, -1)
                else:
                    weightsr = 1.0
                preds = self.output_ref * weightsr
                labels = tf.one_hot(self.label_ref, 2) * weightsr
                losses.append(beta * tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                                labels=labels)))

            # Learning vrms only at the time of reflections
            if gamma > 0:
                mask = tf.cast(self.label_ref, tf.float32)
                num = tf.reduce_sum(mask*self.weights*(self.label_vrms
                                                       - self.output_vrms) ** 2)
                den = tf.reduce_sum(mask*self.weights*self.label_vrms ** 2)
                losses.append(gamma * num / den)

            # Learning interval velocity
            if zeta > 0:
                num = tf.reduce_sum(self.weights*(self.label_vint
                                                  - self.output_vint) ** 2)
                den = tf.reduce_sum(self.weights*self.label_vint ** 2)
                losses.append(zeta * num / den)

            # Minimize interval velocity gradient (blocky inversion)
            if omega > 0:
                num = tf.norm((self.output_vint[:, 1:]
                               - self.output_vint[:, :-1]), ord=1)
                den = tf.norm(self.output_vint, ord=1) / 0.02
                losses.append(omega * num / den)

            # Reconstruction error
            if dec > 0:
                num = tf.norm((self.decoded - self.input_scaled), ord=1)
                den = tf.norm(self.input_scaled, ord=1)
                losses.append(dec * num / den)

            loss = np.sum(losses)

            tf.summary.scalar("loss", loss)
        return loss



def reduce_sum(a, axis=None, keepdims=True):
    if tf.__version__ == '1.2.0':
        return tf.reduce_sum(a, axis=axis, keep_dims=keepdims)
    else:
        return tf.reduce_sum(a, axis=axis, keepdims=keepdims)

def reduce_max(a, axis=None, keepdims=True):
    if tf.__version__ == '1.2.0':
        return tf.reduce_max(a, axis=axis, keep_dims=keepdims)
    else:
        return tf.reduce_max(a, axis=axis, keepdims=keepdims)
