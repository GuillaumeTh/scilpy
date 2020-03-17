
from __future__ import division

import itertools
import logging
import multiprocessing
import os
import sys
import time
import traceback
import warnings

import nibabel as nib
import nibabel.tmpdirs
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.tracking.tissue_tracker import append_to_line


data_file_info = None


def track(tracker, mask, seed, param, compress=False,
          compression_error_threshold=0.1, nbr_processes=1, save_seeds=False):
    """
    Generate a set of streamline from seed, mask and odf files.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    param: dict, tracking parameters, see param.py.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.
    nbr_processes: int, number of sub processes to use.
    pft_tracker: Tracker, tracking object for pft module.
    save_seeds: bool, whether to save the seeds associated to their
        respective streamlines
    Return
    ------
    streamlines: list of numpy.array
    seeds: list of numpy.array
    """
    if param['nbr_streamlines'] == 0:
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                warnings.warn("Cannot determine number of cpus. \
                    returns nbr_processes set to 1.")
                nbr_processes = 1

        param['processes'] = nbr_processes
        if param['processes'] > param['nbr_seeds']:
            nbr_processes = param['nbr_seeds']
            param['processes'] = param['nbr_seeds']
            logging.debug('Setting number of processes to ' +
                          str(param['processes']) +
                          ' since there were less seeds than processes.')
        chunk_id = np.arange(nbr_processes)
        if nbr_processes < 2:
            lines, seeds = get_streamlines(tracker, mask, seed, chunk_id, param, compress,
                                     compression_error_threshold, save_seeds=save_seeds)
        else:

            with nib.tmpdirs.InTemporaryDirectory() as tmpdir:

                # must be better designed for dipy
                # the tracking should not know which data to deal with

                pool = multiprocessing.Pool(nbr_processes)

                lines_per_process, seeds_per_process = zip(*pool.map(
                    _get_streamlines_sub, zip(itertools.repeat(tracker),
                                              itertools.repeat(mask),
                                              itertools.repeat(seed),
                                              chunk_id,
                                              itertools.repeat(param),
                                              itertools.repeat(compress),
                                              itertools.repeat(compression_error_threshold),
                                              itertools.repeat(save_seeds))))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager in order to prevent temporary file deletion
                # errors in Windows
                pool.join()
                lines = np.array([line for line in itertools.chain(*lines_per_process)])
                seeds = np.array([seed for seed in itertools.chain(*seeds_per_process)])
    else:
        if nbr_processes > 1:
            warnings.warn("No multiprocessing implemented while computing " +
                          "a fixed number of streamlines.")
        lines, seeds = get_n_streamlines(tracker, mask, seed, param, compress,
                                   compression_error_threshold, save_seeds=save_seeds)

    return lines, seeds


def _init_sub_process(date_file_name, mmap_mod):
    global data_file_info
    data_file_info = (date_file_name, mmap_mod)
    return


def _get_streamlines_sub(args):
    """
    multiprocessing.pool.map input function.

    Parameters
    ----------
    args : List, parameters for the get_lines(*) function.

    Return
    -------
    lines: list, list of list of 3D positions (streamlines).
    """
    try:
        streamlines, seeds = get_streamlines(*args[0:9])
        return streamlines, seeds
    except Exception as e:
        print("error")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        raise e


def get_n_streamlines(tracker, mask, seeding_mask, param, compress=False,
                      compression_error_threshold=0.1, max_tries=100, save_seeds=True):
    """
    Generate N valid streamlines

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

    Returns
    -------
    lines: list, list of list of 3D positions (streamlines)
    """

    i = 0
    streamlines = []
    seeds = []
    skip = 0
    # Initialize the random number generator, skip,
    # which voxel to seed and the subvoxel random position
    first_seed_of_chunk = np.int32(param['skip'])
    random_generator, indices = seeding_mask.init_pos(param['random'], first_seed_of_chunk)
    while (len(streamlines) < param['nbr_streamlines'] and
           skip < param['nbr_streamlines'] * max_tries):
        if i % 1000 == 0:
            print(str(os.getpid()) + " : " +
                  str(len(streamlines)) + " / " +
                  str(param['nbr_streamlines']))
        seed = seeding_mask.get_next_pos(random_generator,
                                                    indices,
                                                    first_seed_of_chunk + i)
        line = get_line_from_seed(tracker, mask, seed, param)
        if line is not None:
            if compress:
                streamlines.append(compress_streamlines(np.array(line, dtype='float32'),
                                              compression_error_threshold))
            else:
                streamlines.append((np.array(line, dtype='float32')))
            if save_seeds:
                seeds.append(np.asarray(seed, dtype='float32'))

        i += 1
    return streamlines, seeds


def get_streamlines(tracker, mask, seeding_mask, chunk_id, param,
                    compress=False, compression_error_threshold=0.1, save_seeds=True):
    """
    Generate streamlines from all initial positions
    following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    chunk_id: int, chunk id.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

    Returns
    -------
    lines: list, list of list of 3D positions
    """

    streamlines = []
    seeds = []
    # Initialize the random number generator to cover multiprocessing, skip,
    # which voxel to seed and the subvoxel random position
    chunk_size = int(param['nbr_seeds'] / param['processes'])
    skip = param['skip']

    first_seed_of_chunk = chunk_id * chunk_size + skip
    random_generator, indices = seeding_mask.init_pos(param['random'],
                                              first_seed_of_chunk)

    if chunk_id == param['processes'] - 1:
        chunk_size += param['nbr_seeds'] % param['processes']
    for s in range(chunk_size):
        if s % 1000 == 0:
            print(str(os.getpid()) + " : " + str(
                s) + " / " + str(chunk_size))

        seed = seeding_mask.get_next_pos(random_generator,
                                indices,
                                first_seed_of_chunk + s)
        line = get_line_from_seed(tracker, mask, seed, param)
        if len(np.shape(line)) == 1:
            if line is not None:
                for i in np.arange(len(line)):
                    l = line[i]
                    if compress:
                        streamlines.append(compress_streamlines(np.array(l, dtype='float32'),
                                                    compression_error_threshold))
                    else:
                        streamlines.append((np.array(l, dtype='float32')))

                    if save_seeds:
                        seeds.append(np.asarray(seed, dtype='float32'))
        else:
            if line is not None:
                if compress:
                    streamlines.append(compress_streamlines(np.array(line, dtype='float32'),
                                                compression_error_threshold))
                else:
                    streamlines.append((np.array(line, dtype='float32')))
                if save_seeds:
                    seeds.append(np.asarray(seed, dtype='float32'))
    return streamlines, seeds

def get_line_from_seed(tracker, mask, pos, param):
    """
    Generate a streamline from an initial position following the tracking
    paramters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    pos : tuple, 3D position, the seed position.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.

    Returns
    -------
    line: list of 3D positions
    """

    np.random.seed(np.uint32(hash((pos, param['random']))))
    lines = []
    line = []
    others = []
    if tracker.initialize(pos):
        forward = _get_line(tracker, mask, param, True)
        if forward is not None and len(forward) > 0:
            forward_line = forward
            if len(np.shape(forward)) == 1:
                forward_line = forward[0]
                for f in forward[1:]:
                    if len(f) >= param['min_nbr_pts'] and len(f) <= param['max_nbr_pts']:
                        others.append(f)
            line.extend(forward_line)

        if not param['is_single_direction'] and forward is not None and len(line) > 0:
            backward = _get_line(tracker, mask, param, False)
            if backward is not None and len(backward) > 0:
                backward_line = backward
                if len(np.shape(backward)) == 1:
                    backward_line = backward[0]
                    for b in backward[1:]:
                        if len(b) >= param['min_nbr_pts'] and len(b) <= param['max_nbr_pts']:
                            others.append(b)
                line.reverse()
                line.pop()
                line.extend(backward_line)
        else:
            backward = []
        if ((len(line) > 1 and
             forward is not None and
             backward is not None and
             len(line) >= param['min_nbr_pts'] and
             len(line) <= param['max_nbr_pts'])):
            if len(others) > 0:
                lines.append(line)
                lines.extend(others)
            else:
                lines = line
            return lines
        elif (param['is_keep_single_pts'] and
              param['min_nbr_pts'] == 1):
            return [pos]
        return None
    if ((param['is_keep_single_pts'] and
         param['min_nbr_pts'] == 1)):
        return [pos]
    return None


def _get_line(tracker, mask, param, is_forward):
    line = [tracker.init_pos]
    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]

    no_valid_direction_count = 0
    while len(line) < param['max_nbr_pts']:
        new_pos, new_dir, valid_direction_nbr = tracker.propagate(
            line[-1], line_dirs[-1])

        if new_pos is None:
            break

        line, line_dirs, is_finished = append_to_line(tracker, valid_direction_nbr, new_pos, new_dir, line, line_dirs)

        if line is None or line_dirs is None:
            break

        if is_finished:
            return line

    return None

    if line is not None and len(np.shape(line)) == 1:
        for l in line:
            while l is not None and len(l) > 0 and not tracker.isPositionInBound(l[-1]):
                l.pop()
        return line
    else:
        while line is not None and len(line) > 0 and not tracker.isPositionInBound(line[-1]):
            line.pop()
        return line