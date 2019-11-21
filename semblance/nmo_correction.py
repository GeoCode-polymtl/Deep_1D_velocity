"""
A collection of seismic functions to compute semblance, NMO correction and
seismic velocities.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def stack(cmp, times, offsets, velocities):
    """
        Compute the stacked trace of a list of CMP gathers
        
        @params:
        cmps (numpy.ndarray) :  CMP gathers NT X Noffset
        times (numpy.ndarray) : 1D array containing the time
        offsets (numpy.ndarray): 1D array containing the offset of each trace
        velocities (numpy.ndarray): 1D array NT containing the velocities
        
        @returns:
        stacked (numpy.ndarray) : a numpy array NT long containing the stacked
                                  traces of each CMP
        """

    return np.sum(nmo_correction(cmp, times, offsets, velocities), axis=1)

def semblance_gather(cmp, times, offsets, velocities):
    """
    Compute the semblance panel of a CMP gather

    @params:
    cmp (numpy.ndarray) :  CMP gather NT X Noffset
    times (numpy.ndarray) : 1D array containing the time
    offsets (numpy.ndarray): 1D array containing the offset of each trace
    velocities (numpy.ndarray): 1D array containing the test Nv velocities

    @returns:
    semb (numpy.ndarray) : numpy array NTxNv containing semblance
    """
    NT = cmp.shape[0]
    semb = np.zeros([NT, len(velocities)])
    for ii, vel in enumerate(velocities):
        nmo = nmo_correction(cmp, times, offsets, np.ones(NT)*vel)
        semb[:,ii] = semblance(nmo)

    return semb


def nmo_correction(cmp, times, offsets, velocities, stretch_mute=None):
    """
    Compute the NMO corrected CMP gather

    @params:
    cmp (numpy.ndarray) :  CMP gather NT X Noffset
    times (numpy.ndarray) : 1D array containing the time
    offsets (numpy.ndarray): 1D array containing the offset of each trace
    velocities (numpy.ndarray): 1D array containing the test NT velocities
                                in time

    @returns:
    nmo (numpy.ndarray) : array NTxNoffset containing the NMO corrected CMP
    """

    nmo = np.zeros_like(cmp)
    for j, x in enumerate(offsets):
        t = [reflection_time(t0, x, velocities[i]) for i, t0 in enumerate(times)]
        interpolator = CubicSpline(times, cmp[:, j], extrapolate=False)
        amps = np.nan_to_num(interpolator(t), copy=False)
        nmo[:, j] = amps
        if stretch_mute is not None:
            nmo[np.abs((times-t)/(times+1e-10)) > stretch_mute, j] = 0
    return nmo


def reflection_time(t0, x, vnmo):
    """
    Compute the arrival time of a reflecion

    @params:
    t0 (float) :  Two-way travel-time in seconds
    x (float) :  Offset in meters
    vnmo (float): NMO velocity

    @returns:
    t (float): Reflection travel time
    """

    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t

def semblance(nmo_corrected, window=10):
    """
    Compute the semblance of a nmo corrected gather

    @params:
    nmo_corrected (numpy.ndarray) :  NMO corrected CMP gather NT X Noffset
    window (int): Number of time samples to average

    @returns:
    semblance (numpy.ndarray): Array NTx1 containing semblance
    """

    num = np.sum(nmo_corrected, axis=1) ** 2
    den = np.sum(nmo_corrected ** 2, axis=1) + 1e-12
    weights = np.ones(window) / window
    num = np.convolve(num, weights, mode='same')
    den = np.convolve(den, weights, mode='same')
    return num/den



