#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Class to generate seismic models and labels for training.
"""

import numpy as np
import copy
from scipy.signal import gaussian
from scipy.interpolate import interp1d
import argparse
from vrmslearn.ModelParameters import ModelParameters


class ModelGenerator(object):
    """
    Generate a seismic model with the generate_model method and output the
    labels, with generate_labels. As of now, this class generates a 1D layered
    model, and the labels correspond to the rms velocity.
    """

    def __init__(self, model_parameters=ModelParameters()):
        """
        This is the constructor for the class.

        @params:
        model_parameters (ModelParameters)   : A ModelParameter object

        @returns:
        """
        self.pars = model_parameters
        self.vp =None

    def generate_model(self):
        """
        Output the media parameters required for seismic modelling, in this case
        vp, vs and rho. To create 1D model, set pars.flat to True. For 2D dipping
        layer models, set it to False.
        
        @params:
        
        @returns:
        vp (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vp.
        vs (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vs.
        rho (numpy.ndarray) : numpy array (self.pars.NZ, self.pars.NX) for rho
                              values.
        """
        if self.pars.flat:
            vp, vs, rho = generate_random_1Dlayered(self.pars)
        else:
            vp, vs, rho, _, _, _ = generate_random_2Dlayered(self.pars)

        self.vp = copy.copy(vp)
        return vp, vs, rho

    def generate_labels(self):
        """
        Output the labels attached to modelling of a particular dataset. In this
        case, we want to predict vrms from a cmp gather.
        
        @params:
        
        @returns:
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        valid (numpy.ndarray) : numpy array of shape (self.pars.NT, ) with 1
                                before the last reflection, 0 afterwards
        refs (numpy.ndarray) :   Two way travel-times of the reflections
        """
        vp = self.vp[:, 0]
        vrms = calculate_vrms(vp,
                              self.pars.dh,
                              self.pars.Npad,
                              self.pars.NT,
                              self.pars.dt,
                              self.pars.tdelay,
                              self.pars.source_depth)
        refs = generate_reflections_ttime(vp, self.pars)

        # Normalize so the labels are between 0 and 1
        vrms = (vrms - self.pars.vp_min) / (self.pars.vp_max - self.pars.vp_min)
        indt = np.argwhere(refs > 0.1).flatten()[-1]
        valid = np.ones(len(vrms))
        valid[indt:] = 0

        return vrms, valid, refs


def calculate_vrms(vp, dh, Npad, NT, dt, tdelay, source_depth):
    """
    This method inputs vp and outputs the vrms. The global parameters in
    common.py are used for defining the depth spacing, source and receiver
    depth etc. This method assumes that source and receiver depths are same.

    The convention used is that the velocity denoted by the interval
    (i, i+1) grid points is given by the constant vp[i+1].

    @params:
    vp (numpy.ndarray) :  1D vp values in meters/sec.
    dh (float) : the spatial grid size
    Npad (int) : Number of absorbing padding grid points over the source
    NT (int)   : Number of time steps of output
    dt (float) : Time step of the output
    tdelay (float): Time before source peak
    source_depth (float) The source depth in meters


    @returns:
    vrms (numpy.ndarray) : numpy array of shape (NT, ) with vrms
                           values in meters/sec.
    """

    NZ = vp.shape[0]

    # Create a numpy array of depths corresponding to the vp grid locations
    depth = np.zeros((NZ,))
    for i in range(NZ):
        depth[i] = i * dh

    # Create a list of tuples of (relative depths, velocity) of the layers
    # following the depth of the source / receiver depths, till the last layer
    # before the padding zone at the bottom
    last_depth = dh * (NZ - Npad - 1)
    rdepth_vel_pairs = [(d - source_depth, vp[i]) for i, d in enumerate(depth)
                        if d > source_depth and d <= last_depth]
    first_layer_vel = rdepth_vel_pairs[0][1]
    rdepth_vel_pairs.insert(0, (0.0, first_layer_vel))

    # Calculate a list of two-way travel times
    t = [2.0 * (rdepth_vel_pairs[index][0] - rdepth_vel_pairs[index - 1][
        0]) / vel
         for index, (_, vel) in enumerate(rdepth_vel_pairs) if index > 0]
    t.insert(0, 0.0)
    total_time = 0.0
    for i, time in enumerate(t):
        total_time += time
        t[i] = total_time

    # The last time must be 'dt' * 'NT', so adjust the lists 'rdepth_vel_pairs'
    # and 't' by cropping and adjusting the last sample accordingly
    rdepth_vel_pairs = [(rdepth_vel_pairs[i][0], rdepth_vel_pairs[i][1]) for
                        i, time in enumerate(t)
                        if time <= NT * dt]
    t = [time for time in t if time <= NT * dt]
    last_index = len(t) - 1
    extra_distance = (NT * dt - t[last_index]) * rdepth_vel_pairs[last_index][
        1] / 2.0
    rdepth_vel_pairs[last_index] = (
        extra_distance + rdepth_vel_pairs[last_index][0],
        rdepth_vel_pairs[last_index][1])
    t[last_index] = NT * dt

    # Compute vrms at the times in t
    vrms = [first_layer_vel]
    sum_numerator = 0.0
    for i in range(1, len(t)):
        sum_numerator += (t[i] - t[i - 1]) * rdepth_vel_pairs[i][1] * \
                         rdepth_vel_pairs[i][1]
        vrms.append((sum_numerator / t[i]) ** 0.5)

    # Interpolate vrms to uniform time grid
    tgrid = np.asarray(range(0, NT)) * dt
    vrms = np.interp(tgrid, t, vrms)
    vrms = np.reshape(vrms, [-1])
    # Adjust for time delay
    t0 = int(tdelay / dt)
    vrms[t0:] = vrms[:-t0]
    vrms[:t0] = vrms[t0]

    # Return vrms
    return vrms


def generate_random_1Dlayered(pars, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if pars.num_layers == 0:
        nmin = pars.layer_dh_min
        nmax = int(pars.NZ / pars.layer_num_min)
        n_layers = np.random.choice(range(pars.layer_num_min, int(pars.NZ/nmin)))
    else:
        nmin = pars.layer_dh_min
        nmax = int(pars.NZ / pars.layer_num_min)
        n_layers = int(np.clip(pars.num_layers, nmin, nmax))

    NZ = pars.NZ
    NX = pars.NX
    dh = pars.dh
    top_min = int(pars.source_depth / dh + 2 * pars.layer_dh_min)
    layers = (nmin + np.random.rand(n_layers) * (nmax - nmin)).astype(np.int)
    tops = np.cumsum(layers)
    ntos = np.sum(layers[tops <= top_min])
    if ntos > 0:
        layers = np.concatenate([[ntos], layers[tops > top_min]])
    vels = (pars.vp_min
            + np.random.rand() * (pars.vp_max - pars.vp_min - pars.dvmax)
            + np.random.rand(len(layers)) * pars.dvmax)
    ramp = np.abs(np.max(vels) - pars.vp_max) * np.random.rand() + 0.1
    vels = vels + np.linspace(0, ramp, vels.shape[0])
    vels[vels > pars.vp_max] = pars.vp_max
    vels[vels < pars.vp_min] = pars.vp_min
    if pars.marine:
        vels[0] = pars.velwater + (np.random.rand() - 0.5) * 2 * pars.d_velwater
        layers[0] = int(pars.water_depth / pars.dh + (
                np.random.rand() - 0.5) * 2 * pars.dwater_depth / pars.dh)

    vel1d = np.concatenate([np.ones(layers[n]) * vels[n]
                            for n in range(len(layers))])
    if len(vel1d) < NZ:
        vel1d = np.concatenate([vel1d, np.ones(NZ - len(vel1d)) * vel1d[-1]])
    elif len(vel1d) > NZ:
        vel1d = vel1d[:NZ]

    if pars.rho_var:
        rhos = (pars.rho_min
                + np.random.rand() * (
                        pars.rho_max - pars.rho_min - pars.drhomax)
                + np.random.rand(len(layers)) * pars.drhomax)
        ramp = np.abs(np.max(rhos) - pars.rho_max) * np.random.rand() + 0.1
        rhos = rhos + np.linspace(0, ramp, rhos.shape[0])
        rhos[rhos > pars.rho_max] = pars.rho_max
        rhos[rhos < pars.rho_min] = pars.rho_min
        rho1d = np.concatenate([np.ones(layers[n]) * rhos[n]
                                for n in range(len(layers))])
        if len(rho1d) < NZ:
            rho1d = np.concatenate(
                [rho1d, np.ones(NZ - len(rho1d)) * rho1d[-1]])
        elif len(rho1d) > NZ:
            rho1d = rho1d[:NZ]
    else:
        rho1d = vel1d * 0 + pars.rho_default

    vp = np.transpose(np.tile(vel1d, [NX, 1]))
    vs = vp * 0
    rho = np.transpose(np.tile(rho1d, [NX, 1]))

    return vp, vs, rho


def texture_1lay(NZ, NX, lz=2, lx=2):
    """
    Created a random model with bandwidth limited noise.

    @params:
    NZ (int): Number of cells in Z
    NX (int): Number of cells in X
    lz (int): High frequency cut-off size in z
    lx (int): High frequency cut-off size in x
    @returns:

    """

    noise = np.fft.fft2(np.random.random([NZ, NX]))
    noise[0, :] = 0
    noise[:, 0] = 0
    noise[-1, :] = 0
    noise[:, -1] = 0

    iz = lz
    ix = lx
    maskz = gaussian(NZ, iz)
    maskz = np.roll(maskz, [int(NZ / 2), 0])
    maskx = gaussian(NX, ix)
    maskx = np.roll(maskx, [int(NX / 2), 0])
    noise = noise * np.reshape(maskz, [-1, 1])
    noise *= maskx
    noise = np.real(np.fft.ifft2(noise))
    noise = noise / np.max(noise)

    return noise


def generate_reflections_ttime(vp,
                               pars,
                               tol=0.015,
                               window_width=0.45):
    """
    Output the reflection travel time at the minimum offset of a CMP gather

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model
    tol (float): The minimum relative velocity change to consider a reflection
    window_width (float): time window width in percentage of pars.peak_freq

    @returns:

    tabel (numpy.ndarray) : A 2D array with pars.NT elements with 1 at reflecion
                            times +- window_width/pars.peak_freq, 0 elsewhere
    """

    vp = vp[int(pars.source_depth / pars.dh):]
    vlast = vp[0]
    ind = []
    for ii, v in enumerate(vp):
        if np.abs((v - vlast) / vlast) > tol:
            ind.append(ii - 1)
            vlast = v

    if pars.minoffset != 0:
        dt = 2.0 * pars.dh / vp
        t0 = np.cumsum(dt)
        vrms = np.sqrt(t0 * np.cumsum(vp ** 2 * dt))
        tref = np.sqrt(
            t0[ind] ** 2 + pars.minoffset ** 2 / vrms[ind] ** 2) + pars.tdelay
    else:
        ttime = 2 * np.cumsum(pars.dh / vp) + pars.tdelay
        tref = ttime[ind]

    if pars.identify_direct:
        dt = 0
        if pars.minoffset != 0:
            dt = pars.minoffset / vp[0]
        tref = np.insert(tref, 0, pars.tdelay + dt)

    tlabel = np.zeros(pars.NT)
    for t in tref:
        imin = int(t / pars.dt - window_width / pars.peak_freq / pars.dt)
        imax = int(t / pars.dt + window_width / pars.peak_freq / pars.dt)
        if imin <= pars.NT and imax <= pars.NT:
            tlabel[imin:imax] = 1

    return tlabel


def two_way_travel_time(vp, pars):
    """
    Output the two-way travel-time for each cell in vp

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:

    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth, cut to
                          have the same size of t
    t (numpy.ndarray) :  The two-way travel time of each cell

    """
    vpt = vp[int(pars.source_depth / pars.dh):]
    t = 2 * np.cumsum(pars.dh / vpt) + pars.tdelay
    t = t[t < pars.NT * pars.dt]
    vpt = vpt[:len(t)]

    return vpt, t


def interval_velocity_time(vp, pars):
    """
    Output the interval velocity in time

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:

    vint (numpy.ndarray) : The interval velocity in time

    """
    vpt, t = two_way_travel_time(vp, pars)
    interpolator = interp1d(t, vpt,
                            bounds_error=False,
                            fill_value="extrapolate",
                            kind="nearest")
    vint = interpolator(np.arange(0, pars.NT, 1) * pars.dt)

    return vint


def generate_random_2Dlayered(pars, seed=None):
    """
    This method generates a random 2D model, with parameters given in pars.
    Important parameters are:
        Model size:
        -pars.NX : Number of grid cells in X
        -pars.NZ : Number of grid cells in Z
        -pars.dh : Cell size in meters

        Number of layers:
        -pars.num_layers : Minimum number of layers contained in the model
        -pars.layer_dh_min : Minimum thickness of a layer (in grid cell)
        -pars.source_depth: Depth in meters of the source. Velocity above the
                            source is kept constant.

        Layers dip
        -pars.angle_max: Maximum dip of a layer in degrees
        -pars.dangle_max: Maximum dip difference between adjacent layers

        Model velocity
        -pars.vp_max: Maximum Vp velocity
        -pars.vp_min: Minimum Vp velocity
        -pars.dvmax: Maximum velocity difference of two adajcent layers

        Marine survey parameters
        -pars.marine: If True, first layer is water
        -pars.velwater: water velocity
        -pars.d_velwater: variance of water velocity
        -pars.water_depth: Mean water depth
        -pars.dwater_depth: variance of water depth

        Non planar layers
        pars.max_osci_freq: Maximum spatial frequency (1/m) of a layer interface
        pars.min_osci_freq: Minimum spatial frequency (1/m) of a layer interface
        pars.amp_max: Minimum amplitude of the ondulation of the layer interface
        pars.max_osci_nfreq: Maximum number of frequencies of the interface

        Add texture in layers
        pars.texture_zrange
        pars.texture_xrange
        pars.max_texture

    @params:
    pars (str)   : A ModelParameters class containing parameters
                   for model creation.
    seed (str)   : The seed for the random number generator

    @returns:
    vp, vs, rho, vels, layers, angles
    vp (numpy.ndarray)  :  An array containing the vp model
    vs (numpy.ndarray)  :  An array containing the vs model (0 for the moment)
    rho (numpy.ndarray)  :  An array containing the density model
                            (2000 for the moment)
    vels (numpy.ndarray)  : 1D array containing the mean velocity of each layer
    layers (numpy.ndarray)  : 1D array containing the mean thickness of each layer,
                            at the center of the model
    angles (numpy.ndarray)  : 1D array containing slope of each layer
    """

    if seed is not None:
        np.random.seed(seed)

    # Determine the minimum and maximum number of layers
    if pars.num_layers == 0:
        nmin = pars.layer_dh_min
        nmax = int(pars.NZ / pars.layer_num_min)
        if nmin < nmax:
            n_layers = np.random.choice(range(nmin, nmax))
        else:
            n_layers = nmin
    else:
        nmin = pars.layer_dh_min
        nmax = int(pars.NZ / pars.layer_num_min)
        n_layers = int(np.clip(pars.num_layers, nmin, nmax))

    # Generate a random number of layers with random thicknesses
    NZ = pars.NZ
    NX = pars.NX
    dh = pars.dh
    top_min = int(pars.source_depth / dh + 2 * pars.layer_dh_min)
    layers = (nmin + np.random.rand(n_layers) * (nmax - nmin)).astype(np.int)
    tops = np.cumsum(layers)
    ntos = np.sum(layers[tops <= top_min])
    if ntos > 0:
        layers = np.concatenate([[ntos], layers[tops > top_min]])

    # Generate random angles for each layer
    n_angles = len(layers)
    angles = np.zeros(layers.shape)
    angles[1] = -pars.angle_max + np.random.rand() * 2 * pars.angle_max
    for ii in range(2, n_angles):
        angles[ii] = angles[ii - 1] + (
                2.0 * np.random.rand() - 1.0) * pars.dangle_max
        if np.abs(angles[ii]) > pars.angle_max:
            angles[ii] = np.sign(angles[ii]) * pars.angle_max

    # Generate a random velocity for each layer. Velocities are somewhat biased
    # to increase in depth
    vels = (pars.vp_min
            + np.random.rand() * (pars.vp_max - pars.vp_min - pars.dvmax)
            + np.random.rand(len(layers)) * pars.dvmax)
    ramp = np.abs(np.max(vels) - pars.vp_max) * np.random.rand() + 0.1
    vels = vels + np.linspace(0, ramp, vels.shape[0])
    vels[vels > pars.vp_max] = pars.vp_max
    vels[vels < pars.vp_min] = pars.vp_min
    if pars.marine:
        vels[0] = pars.velwater + (np.random.rand() - 0.5) * 2 * pars.d_velwater
        layers[0] = int(pars.water_depth / pars.dh +
                        (
                                np.random.rand() - 0.5) * 2 * pars.dwater_depth / pars.dh)

    # Generate the 2D model, from top layers to bottom
    vel2d = np.zeros([NZ, NX]) + vels[0]
    tops = np.cumsum(layers)
    osci = create_oscillation(pars.max_osci_freq,
                              pars.min_osci_freq,
                              pars.amp_max,
                              pars.max_osci_nfreq, NX)
    texture = texture_1lay(2 * NZ,
                           NX,
                           lz=pars.texture_zrange,
                           lx=pars.texture_xrange)
    for ii in range(0, len(layers) - 1):
        if np.random.rand() < pars.prob_osci_change:
            osci += create_oscillation(pars.max_osci_freq,
                                       pars.min_osci_freq,
                                       pars.amp_max,
                                       pars.max_osci_nfreq, NX)

        texture = texture / np.max(texture) * (
                np.random.rand() + 0.001) * pars.max_texture * vels[ii + 1]
        for jj in range(0, NX):
            # depth of the layer at location x
            dz = int((np.tan(angles[ii + 1] / 360 * 2 * np.pi) * (
                    jj - NX / 2) * dh) / dh)
            # add oscillation component
            if pars.amp_max > 0:
                dz = int(dz + osci[jj])
            # Check if the interface is inside the model
            if 0 < tops[ii] + dz < NZ:
                vel2d[tops[ii] + dz:, jj] = vels[ii + 1]
                if not (pars.marine and ii == 0) and pars.max_texture > 0:
                    vel2d[tops[ii] + dz:, jj] += texture[tops[ii]:NZ - dz, jj]
            elif tops[ii] + dz <= 0:
                vel2d[:, jj] = vels[ii + 1]
                if not (pars.marine and ii == 0) and pars.max_texture > 0:
                    vel2d[:, jj] += texture[:, jj]

    # Output the 2D model
    vel2d[vel2d > pars.vp_max] = pars.vp_max
    vel2d[vel2d < pars.vp_min] = pars.vp_min
    vp = vel2d
    vs = vp * 0
    rho = vp * 0 + 2000

    return vp, vs, rho, vels, layers, angles


def create_oscillation(max_osci_freq, min_osci_freq,
                       amp_max, max_osci_nfreq, Nmax):
    nfreqs = np.random.randint(max_osci_nfreq)
    freqs = np.random.rand(nfreqs) * (
            max_osci_freq - min_osci_freq) + min_osci_freq
    phases = np.random.rand(nfreqs) * np.pi * 2
    amps = np.random.rand(nfreqs)
    x = np.arange(0, Nmax)
    osci = np.zeros(Nmax)
    for ii in range(nfreqs):
        osci += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

    dosci = np.max(osci)
    if dosci > 0:
        osci = osci / dosci * amp_max * np.random.rand()

    return osci


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ND",
        type=int,
        default=1,
        help="Dimension of the model to display"
    )
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    pars = ModelParameters()
    pars.layer_dh_min = 20
    pars.num_layers = 0
    if args.ND == 1:
        vp, vs, rho = generate_random_1Dlayered(pars)
        vp = vp[:, 0]
        vint = interval_velocity_time(vp, pars)
        vrms = calculate_vrms(vp,
                              pars.dh,
                              pars.Npad,
                              pars.NT,
                              pars.dt,
                              pars.tdelay,
                              pars.source_depth)

        plt.plot(vint)
        plt.plot(vrms)
        plt.show()
    else:
        vp, vs, rho = generate_random_2Dlayered(pars)
        plt.imshow(vp)
        plt.show()
