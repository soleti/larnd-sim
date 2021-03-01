#!/usr/bin/env python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from math import ceil
from time import time

import numpy as np
import cupy as cp
import fire
import h5py

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from tqdm import tqdm

from larndsim import consts

logo = """
  _                      _            _
 | |                    | |          (_)
 | | __ _ _ __ _ __   __| |______ ___ _ _ __ ___
 | |/ _` | '__| '_ \ / _` |______/ __| | '_ ` _ \\
 | | (_| | |  | | | | (_| |      \__ \ | | | | | |
 |_|\__,_|_|  |_| |_|\__,_|      |___/_|_| |_| |_|

"""

def cupy_unique_axis0(array):
    # axis is still not supported for cupy.unique, this
    # is a workaround
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cp.lexsort(array.T[::-1])]
    mask        = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0]     = True
    mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

def run_simulation(input_filename,
                   pixel_layout,
                   detector_properties,
                   output_filename='',
                   n_tracks=100000):
    """
    Command-line interface to run the simulation of a pixelated LArTPC

    Args:
        input_filename (str): path of the edep-sim input file
        output_filename (str): path of the HDF5 output file. If not specified
            the output is added to the input file.
        pixel_layout (str): path of the YAML file containing the pixel
            layout and connection details.
        detector_properties (str): path of the YAML file containing
            the detector properties
        n_tracks (int): number of tracks to be simulated
    """

    from cupy.cuda.nvtx import RangePush, RangePop

    RangePush("run_simulation")

    print(logo)
    print("**************************\nLOADING SETTINGS AND INPUT\n**************************")
    print("Pixel layout file:", pixel_layout)
    print("Detector propeties file:", detector_properties)
    print("edep-sim input file:", input_filename)
    RangePush("load_detector_properties")
    consts.load_detector_properties(detector_properties, pixel_layout)
    RangePop()

    RangePush("load_larndsim_modules")
    # Here we load the modules after loading the detector properties
    # maybe can be implemented in a better way?
    from larndsim import quenching, drifting, detsim, pixels_from_track, fee
    RangePop()

    RangePush("load_hd5_file")
    # First of all we load the edep-sim output
    # For this sample we need to invert $z$ and $y$ axes
    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
    RangePop()

    RangePush("slicing_and_swapping")
    tracks = tracks[:n_tracks]

    y_start = np.copy(tracks['y_start'] )
    y_end = np.copy(tracks['y_end'])
    y = np.copy(tracks['y'])

    tracks['y_start'] = np.copy(tracks['z_start'])
    tracks['y_end'] = np.copy(tracks['z_end'])
    tracks['y'] = np.copy(tracks['z'])

    tracks['z_start'] = y_start
    tracks['z_end'] = y_end
    tracks['z'] = y
    RangePop()

    TPB = 256
    BPG = ceil(tracks.shape[0] / TPB)

    print("*******************\nSTARTING SIMULATION\n*******************")
    # We calculate the number of electrons after recombination (quenching module)
    # and the position and number of electrons after drifting (drifting module)
    print("Quenching electrons...",end='')
    start_quenching = time()
    RangePush("quench")
    quenching.quench[BPG,TPB](tracks, consts.birks)
    RangePop()
    end_quenching = time()
    print(f" {end_quenching-start_quenching:.2f} s")

    print("Drifting electrons...",end='')
    start_drifting = time()
    RangePush("drift")
    drifting.drift[BPG,TPB](tracks)
    RangePop()
    end_drifting = time()
    print(f" {end_drifting-start_drifting:.2f} s")
    step = 200
    adc_tot_list = cp.empty((0,fee.MAX_ADC_VALUES))
    adc_tot_ticks_list = cp.empty((0,fee.MAX_ADC_VALUES))
    backtracked_id_tot = cp.empty((0,fee.MAX_ADC_VALUES,5))
    unique_pix_tot = cp.empty((0,2))
    tot_events = 0

    # We divide the sample in portions that can be processed by the GPU
    for itrk in tqdm(range(0, tracks.shape[0], step), desc='Simulating pixels...'):
        selected_tracks = tracks[itrk:itrk+step]

        unique_eventIDs = np.unique(selected_tracks['eventID'])
        event_id_map = np.searchsorted(unique_eventIDs,np.asarray(selected_tracks['eventID']))

        pixels_tracks = np.empty((0,2),dtype=np.int32)
        track_ids = np.array([],dtype=np.int32)

        for it,t in enumerate(selected_tracks):
            this_border = consts.module_borders[int(t["pixel_plane"])]
            start_pixel = (int((t["x_start"] - this_border[0][0]) // consts.pixel_size[0] + consts.n_pixels[0]*t["pixel_plane"]),
                        int((t["y_start"] - this_border[1][0]) // consts.pixel_size[1]))
            end_pixel = (int((t["x_end"] - this_border[0][0]) // consts.pixel_size[0] + consts.n_pixels[0]*t["pixel_plane"]),
                        int((t["y_end"] - this_border[1][0]) // consts.pixel_size[1]))
            pixels = pixels_from_track.get_active_pixels(start_pixel[0], start_pixel[1],
                                                         end_pixel[0], end_pixel[1])
            neighboring_pixels = pixels_from_track.get_neighboring_pixels(np.array(pixels,dtype=np.int32),2)
            pixels_tracks = np.vstack((pixels_tracks,neighboring_pixels))
            track_ids = np.append(track_ids,[it]*len(neighboring_pixels))

        unique_pix = np.unique(pixels_tracks,axis=0)

        max_length = np.array([0])
        track_starts = np.empty(selected_tracks.shape[0])
        threadsperblock = 128
        blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
        detsim.time_intervals[blockspergrid,threadsperblock](track_starts, max_length,  event_id_map, selected_tracks)

        pixels_tracks_time = np.zeros((pixels_tracks.shape[0],max_length[0]))
        TPB = 1,512
        BPG = ceil(pixels_tracks_time.shape[0] / TPB[0]), ceil(pixels_tracks_time.shape[1] / TPB[1])
        detsim.tracks_current[BPG,TPB](pixels_tracks_time, pixels_tracks, selected_tracks, track_ids)

        TPB = 1,512
        BPG = ceil(pixels_tracks_time.shape[0] / TPB[0]), ceil(pixels_tracks_time.shape[1] / TPB[1])
        pixel_index_map = np.ascontiguousarray(np.where((unique_pix==pixels_tracks[:,None]).all(-1))[1])
        pixels_signals = np.zeros((len(unique_pix), len(consts.time_ticks)*len(unique_eventIDs)*2))
        detsim.sum_pixel_signals[BPG,TPB](pixels_signals, pixels_tracks_time, track_starts, pixel_index_map, track_ids)


if __name__ == "__main__":
    fire.Fire(run_simulation)
