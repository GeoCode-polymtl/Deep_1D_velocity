#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to generate the labels (seismic data)
"""

from vrmslearn.ModelGenerator import ModelGenerator, interval_velocity_time
from vrmslearn.ModelParameters import ModelParameters
from SeisCL.SeisCL import SeisCL
import numpy as np
import os
import h5py as h5
import fnmatch
from multiprocessing import Process, Queue, Event, Value
from shutil import rmtree
import fnmatch
from  scipy.signal import convolve2d

def gaussian(f0, t, o, amp=1.0, order=2):

    x = np.pi * f0 * (t + o)
    e = amp * np.exp(-x ** 2)
    if order == 1:
        return e*x
    elif order == 2:
        return (1.0 - 2.0 * x ** 2) * e
    elif order == 3:
        return 2.0 * x * (2.0 * x ** 2 - 3.0) * e
    elif order == 4:
        return (-8.0 * x ** 4 + 24.0 * x ** 2 - 6.0) * e
    elif order == 5:
        return 4.0 * x * (4.0 * x ** 2 - 20.0 * x ** 2 + 15.0) * e
    elif order == 6:
        return -4.0 * (8.0 * x ** 6 - 60.0 * x ** 4 + 90.0 * x ** 2 - 15.0) * e

def morlet(f0, t, o , amp=1.0, order=5):
    x = f0 * (t + o)
    return amp * np.cos(x*order) * np.exp(- x ** 2)


def shift_trace(signal, phase):
    S = np.fft.fft(signal)
    NT = len(signal)
    S[1:NT//2] *= 2.0
    S[NT // 2+1:] *=0
    s = np.fft.ifft(S)
    return np.real(s) * np.cos(phase) + np.imag(s) * np.sin(phase)

class SeismicGenerator(object):
    """
    Class to generate seismic data with SeisCL and output an example to build
    a seismic dataset for training.
    """
    def __init__(self,
                 model_parameters=ModelParameters(),
                 gpus=[]):
        """
        This is the constructor for the class.

        @params:
        model_parameters (ModelParameters)   : A ModelParameter object
        gpus (list): A list of GPUs not to use for computation
        wavefuns (list)L A list of wave function generator to source generation

        @returns:
        """

        self.pars = model_parameters
        self.F = SeisCL()

        # Overload to generate other kind of models
        self.model_generator = ModelGenerator(model_parameters=model_parameters)

        self.init_F(gpus)
        self.image_size = [int(np.ceil(self.pars.NT/self.pars.resampling)),
                           self.F.rec_pos.shape[1]]

        allwavefuns= [lambda f0, t, o: gaussian(f0, t, o, order=1),
                      lambda f0, t, o: gaussian(f0, t, o, order=2),
                      lambda f0, t, o: gaussian(f0, t, o, order=3),
                      lambda f0, t, o: morlet(f0, t, o, order=2),
                      lambda f0, t, o: morlet(f0, t, o, order=3),
                      lambda f0, t, o: morlet(f0, t, o, order=4)]


        self.wavefuns = [allwavefuns[ii] for ii in model_parameters.wavefuns]

        self.files_list = {}

    def init_F(self, gpus=[]):
        """
        This method initializes the variable 'self.F', which is used for forward
        modeling using the SeisCL engine. We assume here a 1D vp model and we
        position a source at the top of the model, centered in the x direction.

        @params:
        gpus (list) :   A list of GPU ids not to use

        @returns:
        """
        # Initialize the modeling engine
        self.F.csts['N'] = np.array([self.pars.NZ, self.pars.NX])
        self.F.csts['ND'] = 2
        self.F.csts['dh'] = self.pars.dh                          # Grid spacing
        self.F.csts['nab'] = self.pars.Npad                  # Set padding cells
        self.F.csts['dt'] = self.pars.dt                        # Time step size
        self.F.csts['NT'] = self.pars.NT                      # Nb of time steps
        self.F.csts['f0'] = self.pars.peak_freq               # Source frequency
        self.F.csts['seisout'] = 2                             # Output pressure
        self.F.csts['freesurf'] = int(self.pars.fs)               # Free surface
        self.F.csts['no_use_GPUs'] = np.array(gpus)
        self.F.csts['pref_device_type'] = self.pars.device_type

        if self.pars.flat:
            # Add a source in the middle
            sx = np.arange(self.pars.NX / 2, 1 + self.pars.NX / 2) * self.pars.dh
        else:
            if self.pars.train_on_shots:
                l1 = self.pars.Npad + 1
                if self.pars.gmin and self.pars.gmin < 0:
                    l1 += -self.pars.gmin
                l2 = self.pars.NX - self.pars.Npad
                if self.pars.gmax and self.pars.gmax > 0:
                    l2 += -self.pars.gmax

                sx = np.arange(l1, l2, self.pars.ds) * self.pars.dh
            else:
                # We need to compute the true CMP as layers have a slope.
                # We compute one CMP, with multiple shots with 1 receiver
                sx = np.arange(self.pars.Npad + 1,
                               self.pars.NX - self.pars.Npad,
                               self.pars.dg) * self.pars.dh
        sz = sx * 0 + self.pars.source_depth
        sid = np.arange(0, sx.shape[0])

        self.F.src_pos = np.stack([sx,
                                   sx * 0,
                                   sz,
                                   sid,
                                   sx * 0 + self.pars.sourcetype], axis=0)
        self.F.src_pos_all = self.F.src_pos
        self.F.src = np.empty((self.F.csts['NT'], 0))

        def generate_wavelet():
            t = np.arange(0, self.pars.NT) * self.pars.dt
            fmin = self.pars.peak_freq - self.pars.df
            fmax = self.pars.peak_freq + self.pars.df
            f0 = np.random.rand(1)*(fmax-fmin) + fmin
            phase = np.random.rand(1) * np.pi
            fun = np.random.choice(self.wavefuns)
            tdelay = -self.pars.tdelay
            src = fun(f0, t, tdelay)
            src = shift_trace(src, phase)

            return src

        self.F.wavelet_generator = generate_wavelet

        # Add receivers
        if self.pars.flat or self.pars.train_on_shots:
            if self.pars.gmin:
                gmin = self.pars.gmin
            else:
                gmin = -(self.pars.NX - 2 *self.pars.Npad) // 2
            if self.pars.gmax:
                gmax = self.pars.gmax
            else:
                gmax = (self.pars.NX - 2 *self.pars.Npad) //2

            gx0 = np.arange(gmin, gmax, self.pars.dg) * self.pars.dh
            gx = np.concatenate([s + gx0 for s in sx], axis=0)
            gsid = np.concatenate([s + gx0 * 0 for s in sid], axis=0)

        else:
            # One receiver per source, with the middle point at NX/2
            gx = (self.pars.NX - sx/self.pars.dh) * self.pars.dh
            gsid = sid
        gz = gx * 0 + self.pars.receiver_depth
        gid = np.arange(0, len(gx))

        self.F.rec_pos = np.stack([gx,
                                   gx * 0,
                                   gz,
                                   gsid,
                                   gid,
                                   gx * 0 + 2,
                                   gx * 0,
                                   gx * 0], axis=0)
        self.F.rec_pos_all = self.F.rec_pos


    def compute_example(self, workdir):
        """
        This method generates one example, which contains the vp model, vrms,
        the seismic data and the valid vrms time samples.

        @params:
        workdir (str)   : A string containing the working direction of SeisCL

        @returns:
        data (numpy.ndarray)  : Contains the modelled seismic data
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        vp (numpy.ndarray)    : numpy array (self.pars.NZ, self.pars.NX) for vp.
        valid (numpy.ndarray) : numpy array (self.pars.NT, )containing the time
                                samples for which vrms is valid
        tlabels (numpy.ndarray) : numpy array (self.pars.NT, ) containing the
                                  if a sample is a primary reflection (1) or not
        """
        vp, vs, rho = self.model_generator.generate_model()
        vrms, valid, tlabels = self.model_generator.generate_labels()
        self.F.set_forward(self.F.src_pos[3, :],
                           {'vp': vp, 'vs': vs, 'rho': rho},
                           workdir,
                           withgrad=False)
        self.F.execute(workdir)
        data = self.F.read_data(workdir)
        data = data[0][::self.pars.resampling, :]
        vrms = vrms[::self.pars.resampling]
        valid = valid[::self.pars.resampling]
        tlabels = tlabels[::self.pars.resampling]

        return data, vrms, vp[:,0], valid, tlabels

    def write_example(self, n, savedir, data, vrms, vp, valid, tlabels,
                      filename=None):
        """
        This method writes one example in the hdf5 format

        @params:
        savedir (str)   :       A string containing the directory in which to
                                save the example
        data (numpy.ndarray)  : Contains the modelled seismic data
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        vp (numpy.ndarray)    : numpy array (self.pars.NZ, self.pars.NX) for vp.
        valid (numpy.ndarray) : numpy array (self.pars.NT, )containing the time
                                samples for which vrms is valid
        tlabels (numpy.ndarray) : numpy array (self.pars.NT, ) containing the
                                  if a sample is a primary reflection (1) or not

        @returns:
        n (int)               : The number of examples in the directory
        """
        
        if filename is None:
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            #n = len(fnmatch.filter(os.listdir(savedir), 'example_*'))
            pid = os.getpid()
            filename= savedir + "/example_%d_%d" % (n, pid)
        else:
            filename = savedir + "/" +filename

        file = h5.File(filename, "w")
        file["data"] = data
        file["vrms"] = vrms
        file["vp"] = vp
        file["valid"] = valid
        file["tlabels"] = tlabels
        file.close()

        return n

    def read_example(self, savedir, filename=None):
        """
        This method retrieve one example written in the hdf5 format

        @params:
        savedir (str)   :       A string containing the directory in which to
                                read the example
        filename (str)   :      If provided, the file name of the example to read

        @returns:
        data (numpy.ndarray)  : Contains the modelled seismic data
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        vint (numpy.ndarray)   : numpy array (self.pars.NT,) containing interval
                                 velocity
        valid (numpy.ndarray) : numpy array (self.pars.NT, )containing the time
                                samples for which vrms is valid
        tlabels (numpy.ndarray) : numpy array (self.pars.NT, ) containing the
                                  if a sample is a primary reflection (1) or not

        """
        if filename is None:

            if type(savedir) is list:
                savedir = np.random.choice(savedir, 1)[0]
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            if savedir not in self.files_list:
                files = fnmatch.filter(os.listdir(savedir), 'example_*')
                self.files_list[savedir] = files
            else:
                files = self.files_list[savedir]
            if not files:
                raise FileNotFoundError()
            filename = savedir + "/" + np.random.choice(files, 1)[0]
        else:
            filename = savedir + "/" + filename

        file = h5.File(filename, "r")
        data = file['data'][:]
        vrms = file['vrms'][:]
        vp = file['vp'][:]
        if self.pars.random_time_scaling:
            data = random_time_scaling(data, self.pars.dt * self.pars.resampling)
        if self.pars.mute_dir:
            data = mute_direct(data, vp[0], self.pars)
        if self.pars.random_static:
            data = random_static(data, self.pars.random_static_max)
        if self.pars.random_noise:
            data = random_noise(data, self.pars.random_noise_max)
        if self.pars.mute_nearoffset:
            data = mute_nearoffset(data, self.pars.mute_nearoffset_max)


        vint = interval_velocity_time(vp, self.pars)[::self.pars.resampling]
        vint = (vint - self.pars.vp_min) / (self.pars.vp_max - self.pars.vp_min)
        valid = file['valid'][:]
        tlabels = file['tlabels'][:]
        if self.pars.mask_firstvel:
            ind0 = np.nonzero(tlabels)[0][0]
            valid[0:ind0] *= 0.05
        file.close()

#        return data.tolist(), vrms.tolist(), vint.tolist(), valid.tolist(), tlabels.tolist()
#    
        return data, vrms, vint, valid, tlabels

    def aggregate_examples(self, batch):
        """
        This method aggregates a batch of examples

        @params:
        batch (lsit):       A list of numpy arrays that contain a list with
                            all elements of of example.

        @returns:
        batch (numpy.ndarray): A list of numpy arrays that contains all examples
                                 for each element of a batch.

        """
        data = np.stack([el[0] for el in batch])
        data = np.expand_dims(data, axis=-1)
        vrms = np.stack([el[1] for el in batch])
        vint = np.stack([el[2] for el in batch])
        weights = np.stack([el[3] for el in batch])
        tlabels = np.stack([el[4] for el in batch])

        return [data, vrms, weights, tlabels, vint]


def generate_dataset(pars: ModelParameters,
                     savepath: str,
                     nexamples: int,
                     seed: int=None,
                     nthread: int=3,
                     workdir: str="./workdir"):
    """
    This method creates a dataset. If multiple threads or processes generate
    the dataset, it may not be totally reproducible due to a different
    random seed attributed to each process or thread.

    @params:
    pars (ModelParameter): A ModelParamter object
    savepath (str)   :     Path in which to create the dataset
    nexamples (int):       Number of examples to generate
    seed (int):            Seed for random model generator
    nthread (int):         Number of processes used to generate examples
    workdir (str):         Name of the directory for temporary files

    @returns:

    """

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    generators = []
    for jj in range(nthread):
        this_workdir = workdir + "_" + str(jj)
        if seed is not None:
            thisseed = seed * (jj + 1)
        else:
            thisseed = seed
        thisgen = DatasetGenerator(pars,
                                   savepath,
                                   this_workdir,
                                   nexamples,
                                   [],
                                   seed=thisseed)
        thisgen.start()
        generators.append(thisgen)
    for gen in generators:
        gen.join()

class DatasetGenerator(Process):
    """
    This class creates a new process to generate seismic data.
    """

    def __init__(self,
                 parameters,
                 savepath: str,
                 workdir: str,
                 nexamples: int,
                 gpus: list,
                 seed: int=None):
        """
        Initialize the DatasetGenerator

        @params:
        parameters (ModelParameter): A ModelParamter object
        savepath (str)   :     Path in which to create the dataset
        workdir (str):         Name of the directory for temporary files
        nexamples (int):       Number of examples to generate
        gpus (list):           List of gpus not to use.
        seed (int):            Seed for random model generator

        @returns:
        """
        super().__init__()

        self.savepath = savepath
        self.workdir = workdir
        self.nexamples = nexamples
        self.parameters = parameters
        self.gpus = gpus
        self.seed = seed
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
        try:
            parameters.save_parameters_to_disk(savepath
                                               + "/model_parameters.hdf5")
        except OSError:
            pass

    def run(self):
        """
        Start the process to generate data
        """
        n = len(fnmatch.filter(os.listdir(self.savepath), 'example_*'))
        gen = SeismicGenerator(model_parameters=self.parameters,
                               gpus=self.gpus)
        if self.seed is not None:
            np.random.seed(self.seed)
        
        while n < self.nexamples:
            n = len(fnmatch.filter(os.listdir(self.savepath), 'example_*'))
            if self.seed is None:
                np.random.seed(n)
            data, vrms, vp, valid, tlabels = gen.compute_example(self.workdir)
            try:
                gen.write_example(n, self.savepath, data, vrms, vp, valid, tlabels)
                if n % 100 == 0:
                    print("%f of examples computed" % (float(n)/self.nexamples))
            except OSError:
                pass
        if os.path.isdir(self.workdir):
            rmtree(self.workdir)


def mask_batch(batch,
              mask_fraction,
              mask_time_frac):

    for ii, el in enumerate(batch):
        data = el[0]
        NT = data.shape[0]
        ng = data.shape[1]

        #Mask time and offset
        frac = np.random.rand() * mask_time_frac
        twindow = int(frac * NT)
        owindow = int(frac * ng / 2)
        batch[ii][0][-twindow:,:] = 0
        batch[ii][0][:,:owindow] = 0
        batch[ii][0][:, -owindow:] = 0

        #take random subset of traces
        ntokill = int(np.random.rand()*mask_fraction*ng*frac)
        tokill = np.random.choice(np.arange(owindow, ng-owindow), ntokill, replace=False)
        batch[ii][0][:, tokill] = 0

        batch[ii][3][-twindow:] = 0



    return batch


def mute_direct(data, vp0, pars, offsets=None):

    wind_length = int(2 / pars.peak_freq / pars.dt / pars.resampling)
    taper = np.arange(wind_length)
    taper = np.sin(np.pi * taper / (2 * wind_length - 1)) ** 2
    NT = data.shape[0]
    ng = data.shape[1]
    if offsets is None:
        if pars.gmin is None or pars.gmax is None:
            offsets = (np.arange(0, ng) - (ng) / 2) * pars.dh * pars.dg
        else:
            offsets = (np.arange(pars.gmin, pars.gmax, pars.dg)) * pars.dh



    for ii, off in enumerate(offsets):
        tmute = int(( np.abs(off) / vp0 + 1.5 * pars.tdelay) / pars.dt / pars.resampling)
        if tmute <= NT:
            data[0:tmute, ii] = 0
            mute_max = np.min([tmute + wind_length, NT])
            nmute = mute_max - tmute
            data[tmute:mute_max, ii] = data[tmute:mute_max, ii] * taper[:nmute]
        else:
            data[:, ii] = 0

    return data

def random_static(data, max_static):
    
    ng = data.shape[1]
    shifts = (np.random.rand(ng) - 0.5) * max_static * 2
    for ii in range(ng):
        data[:, ii] = np.roll(data[:, ii], int(shifts[ii]), 0)
    return data

def random_noise(data, max_amp):
    
    max_amp = max_amp * np.max(data) * 2.0
    data = data + (np.random.rand(data.shape[0],data.shape[1] ) - 0.5) * max_amp
    return data

def mute_nearoffset(data, max_off):

    data[:,:np.random.randint(max_off)] *= 0
    return data

def random_filt(data, filt_length):
    filt_length = int((np.random.randint(filt_length)//2)*2 +1)
    filt = np.random.rand(filt_length, 1)
    data = convolve2d(data, filt, 'same')
    return data

def random_time_scaling(data, dt, emin=-2.0, emax=2.0, scalmax=None):
    t = np.reshape(np.arange(0, data.shape[0]) * dt, [data.shape[0], 1])
    e = np.random.rand() * (emax - emin) + emin
    scal = (t+1e-6) ** e
    if scalmax is not None:
        scal[scal>scalmax] = scalmax
    return data * scal


