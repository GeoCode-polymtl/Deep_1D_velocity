#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing of real data available publicly at:
https://cmgds.marine.usgs.gov/fan_info.php?fan=1978-015-FA
"""

import urllib.request
import os
import segyio
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import scipy.ndimage as ndimage
import math

if __name__ == "__main__":
    
    """
        __________________Download the data______________________
    """
    
    datapath = "./USGS_line32"

    files = {"32obslog.pdf": "http://cotuit.er.usgs.gov/files/1978-015-FA/NL/001/01/32-obslogs/32obslog.pdf",
             "report.pdf": "https://pubs.usgs.gov/of/1995/0027/report.pdf",
             "CSDS32_1.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/39/CSDS32_1.SGY"}


    dfiles = {"U32A_01.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_01.SGY",
            "U32A_02.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_02.SGY",
            "U32A_03.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_03.SGY",
            "U32A_04.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_04.SGY",
            "U32A_05.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_05.SGY",
            "U32A_06.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_06.SGY",
            "U32A_07.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_07.SGY",
            "U32A_08.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_08.SGY",
            "U32A_09.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_09.SGY"}
            # "U32A_10.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_10.SGY",
            # "U32A_11.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_11.SGY",
            # "U32A_12.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_12.SGY",
            # "U32A_13.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_13.SGY",
            # "U32A_14.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_14.SGY",
            # "U32A_15.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_15.SGY",
            # "U32A_16.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_16.SGY",
            # "U32A_17.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_17.SGY",
            # "U32A_18.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_18.SGY",
            # "U32A_19.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_19.SGY",
            # "U32A_20.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_20.SGY",
            # "U32A_21.SGY": "http://cotuit.er.usgs.gov/files/1978-015-FA/SE/001/18/U32A_21.SGY"}
    
    
    fkeys = sorted(list(dfiles.keys()))
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    for file in files:
        if not os.path.isfile(datapath + "/" + file):
            urllib.request.urlretrieve(files[file], datapath + "/" + file)

    for file in dfiles:
        if not os.path.isfile(datapath + "/" + file):
            print(file)
            urllib.request.urlretrieve(dfiles[file], datapath + "/" + file)


    """
    __________________Read the segy into numpy______________________
    """
    data = []
    fid = []
    cid = []
    NT = 3071
    for file in fkeys:
        print(file)
        with segyio.open(datapath + "/" + file, "r", ignore_geometry=True) as segy:
            fid.append([segy.header[trid][segyio.TraceField.FieldRecord]
                                          for trid in range(segy.tracecount)])
            cid.append([segy.header[trid][segyio.TraceField.TraceNumber]
                        for trid in range(segy.tracecount)])
            data.append(np.transpose(np.array([segy.trace[trid]
                                          for trid in range(segy.tracecount)]))[:NT,:])


    """
    __________________Remove bad shots ______________________
    """
    #correct fid
    if len(fid) > 16:
        fid[16] = [id if id < 700 else id+200 for id in fid[16]]
    if len(fid) > 6:
        fid[6] = fid[6][:12180]
        cid[6] = cid[6][:12180]
        data[6] = data[6][:, :12180]
    if len(fid) > 7:
        fid[7] = fid[7][36:]
        cid[7] = cid[7][36:]
        data[7] = data[7][:, 36:]
    if len(fid) > 2: #repeated shots between files 03 and 04
        fid[2] = fid[2][:8872]
        cid[2] = cid[2][:8872]
        data[2] = data[2][:, :8872]
    fid = np.concatenate(fid)
    cid = np.concatenate(cid)
    data = np.concatenate(data, axis=1)

    #recnoSpn = InterpText()
    #recnoSpn.read('recnoSpn.txt')

    #recnoDelrt = InterpText()
    #recnoDelrt.read('recnoDelrt.txt')

    prev_fldr=-9999
    fldr_bias=0
    shot = 0 * cid -1
    delrt = 0 * cid -1

    notshots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 211, 213, 225, 279,
                335, 387, 400, 493, 528, 553, 561, 571,
                668, 669, 698, 699, 700, 727, 728, 780, 816, 826, 1073, 1219,
                1253, 1254, 1300, 1301, 1418, 1419, 1527, 1741, 2089, 2170,
                2303, 2610, 2957, 2980, 3021, 3104, 3167, 3223, 3268, 3476,
                3707, 3784, 3831, 3934, 4051, 4472, 4671, 4757, 4797]

    for ii in range(fid.shape[0]):
        fldr = fid[ii]
        tracf = cid[ii]

        if fldr < prev_fldr:
            fldr_bias += 1000

        prev_fldr = fldr

        fldr += fldr_bias
        if fldr not in notshots:
            shot[ii] = 6102 - fldr
        
        # The time 0 of different files changes. We prepad with zero so that all
        # shots begin at time 0
        if fldr < 15:
            delrt[ii] = 4000
        elif fldr < 20:
            delrt[ii] = 5000
        elif fldr < 1043:
            delrt[ii] = 4000
        elif fldr < 1841:
            delrt[ii] = 3000
        elif fldr < 2199:
            delrt[ii] = 2000
        elif fldr < 2472:
            delrt[ii] = 1000
        else:
            delrt[ii] = 0

    valid = shot > 0
    shot = shot[valid]
    delrt = delrt[valid]
    data = data[:, valid]


    plt.plot(shot)
    plt.show()

    dt = 4 # time step, milliseconds
    for ii in range(data.shape[1]):
        data[:, ii] = np.concatenate([np.zeros(int(delrt[ii]/dt)), data[:,ii]])[:NT]

    # Open the hdf5 file in which to save the pre-processed data
    savefile = h5.File("survey.hdf5", "w")
    savefile["data"] = data
    
    """
    ________________________Trace interpolation____________________________
    """

    #From the observer log, we get the acquisition parameters:
    ds = 50  #shot point spacing
    dg1 = 100  #geophone spacing for channels 1-24
    dg2 = 50  #geophone spacing for channels 25-48
    vwater = 1533
    ns = int(data.shape[1]/48)
    ng = 72
    dg = 50
    nearoff = 470 #varies for several shots, we take the most common value

    data_i = np.zeros([data.shape[0], ns*ng])
    t0off = 2*np.sqrt((nearoff / 2)**2 +3000**2)/vwater
    for ii in range(ns):
        data_i[:, ng*ii:ng*ii+23] = data[:, ii*48:ii*48+23]
        data_roll = data[:, ii*48+23:(ii+1) * 48]
        n = data_roll.shape[1]
        for jj in range(n):
            toff = 2 * np.sqrt(((nearoff + dg1 * (n - jj)) / 2) ** 2 + 3000 ** 2) / vwater - t0off
            data_roll[:, jj] = np.roll(data_roll[:, jj], -int(toff / 0.004))
        data_roll = ndimage.zoom(data_roll, [1, 2], order=1)
        n = data_roll.shape[1]
        for jj in range(n):
            toff = 2 * np.sqrt(
                ((nearoff + dg2 * (n - jj)) / 2) ** 2 + 3000 ** 2) / vwater - t0off
            data_roll[:, jj] = np.roll(data_roll[:, jj], int(toff / 0.004))
        data_i[:, ng * ii + 23:ng * (ii + 1)] = data_roll[:, :-1]

    savefile['data_i'] = data_i

    """
    ________________________Resort accorging to CMP____________________________
    """
    ns = int(data_i.shape[1]/72)
    shots = np.arange(nearoff + ng*dg, nearoff + ng*dg + ns * ds, ds)
    recs = np.concatenate([np.arange(0, 0 + ng * dg, dg) + n*ds for n in range(ns)], axis=0)
    shots = np.repeat(shots, ng)
    cmps = ((shots + recs)/2 / 50).astype(int) * 50
    offsets = shots - recs

    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    unique_cmps, counts = np.unique(cmps, return_counts=True)
    firstcmp = unique_cmps[np.argmax(counts == 72)]
    lastcmp = unique_cmps[-np.argmax(counts[::-1] == 72)-1]
    ind1 = np.argmax(cmps == firstcmp)
    ind2 = np.argmax(cmps > lastcmp)
    ntraces = cmps[ind1:ind2].shape[0]
    data_cmp = np.zeros([data_i.shape[0], ntraces])

    n = 0
    for ii, jj in enumerate(ind):
        if ii >= ind1 and ii < ind2:
            data_cmp[:, n] = data_i[:, jj]
            n += 1

    savefile['data_cmp'] = data_cmp
    savefile.close()

    """
    ________________________Plots for quality control___________________________
    """
    # Plot some CMP gather
    clip = 0.05
    vmax = np.max(data_cmp[:,0]) * clip
    vmin = -vmax
    plt.imshow(data_cmp[:, :200],
               interpolation='bilinear',
                cmap=plt.get_cmap('Greys'),
                vmin=vmin, vmax=vmax,
                aspect='auto')

    plt.show()

    # Constant offset plot
    clip = 0.05
    vmax = np.max(data_cmp[:,0]) * clip
    vmin = -vmax
    plt.imshow(data_cmp[:, ::72],
               interpolation='bilinear',
               cmap=plt.get_cmap('Greys'),
               vmin=vmin, vmax=vmax,
               aspect='auto')

    plt.show()

