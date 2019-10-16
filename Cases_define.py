#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Defines parameters for different cases
"""

from vrmslearn.ModelParameters import ModelParameters

def Case_small():
    return ModelParameters()

def Case_article(noise=0):
    pars = ModelParameters()
    pars.layer_dh_min = 5
    pars.layer_num_min = 48
    
    pars.dh = 6.25
    pars.peak_freq = 26
    pars.df = 5
    pars.wavefuns = [0, 1]
    pars.NX = 692*2
    pars.NZ = 752*2
    pars.dt = 0.0004
    pars.NT = int(8.0 / pars.dt)
    pars.resampling = 10
    
    pars.dg = 8
    pars.gmin = int(470 / pars.dh)
    pars.gmax = int((470 + 72 * pars.dg * pars.dh) / pars.dh)
    pars.minoffset = 470
    
    pars.vp_min = 1300.0  # maximum value of vp (in m/s)
    pars.vp_max = 4000.0  # minimum value of vp (in m/s)
    
    pars.marine = True
    pars.velwater = 1500
    pars.d_velwater = 60
    pars.water_depth = 3500
    pars.dwater_depth = 1000
    
    pars.fs = False
    pars.source_depth = (pars.Npad + 4) * pars.dh
    pars.receiver_depth = (pars.Npad + 4) * pars.dh
    pars.identify_direct = False
    
    pars.mute_dir = True
    if noise == 1:
        pars.random_static = True
        pars.random_static_max = 1
        pars.random_noise = True
        pars.random_noise_max = 0.02
    #    pars.mute_nearoffset = True
    #    pars.mute_nearoffset_max = 10

    return pars


