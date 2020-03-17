#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import logging
import math
import os
import time

import dipy.core.geometry as gm
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.utils import add_sh_basis_args
from scilpy.tracking.dataset import Dataset
from scilpy.tracking.local_tracking import track
from scilpy.tracking.mask import BinaryMask
from scilpy.tracking.seed import Seed
from scilpy.tracking.tools import (compute_average_streamlines_length)
from scilpy.tracking.tracker import (probabilisticTracker)
from scilpy.tracking.trackingField import SphericalHarmonicField


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Local streamline HARDI tractography. The tracking is done'
        + ' inside a binary mask. Streamlines greater than minL and shorter '
        + 'than maxL are outputted. The tracking direction is chosen in the '
        + 'aperture cone defined by the previous tracking direction and the '
        + 'angular constraint. The relation between theta and the curvature '
        + "is theta=2*arcsin(step_size/(2*R)). \n\nAlgo 'det': the maxima "
        + "of the spherical "
        + "function (SF) the most closely aligned to the previous direction."
        + "\nAlgo 'prob': a "
        + 'direction drawn from the empirical distribution function defined '
        + 'from the SF. \nDefault parameters as in [1].',
        epilog='References: [1] Girard, G., Whittingstall K., Deriche, R., and'
        + ' Descoteaux, M. (2014). Towards quantitative connectivity analysis:'
        + ' reducing tractography biases. Neuroimage, 98, 266-278.')
    p._optionals.title = "Options and Parameters"

    p.add_argument(
        'sh_file', action='store', metavar='sh_file', type=str,
        help="Spherical Harmonic file. Data must be aligned with \nseed_file" +
        " (isotropic resolution, nifti, see --sh_basis).")
    p.add_argument(
        'seed_file', action='store', metavar='seed_file', type=str,
        help="Seeding mask (isotropic resolution, nifti).")
    p.add_argument(
        'atlas', action='store', metavar='atlas_file', type=str,
        help="Atlas (isotropic resolution, nifti).")
    p.add_argument(
        'config', action='store', metavar='config_file', type=str,
        help="CONFIG (isotropic resolution, nifti).")
    p.add_argument(
        'output_file', action='store', metavar='output_file', type=str,
        help="Streamline output file (must be trk or tck).")

    add_sh_basis_args(p)
    # p.add_argument(
    #     '--algo', dest='algo', action='store', metavar='ALGO', type=str,
    #     default='det', choices=['det', 'prob'],
    #     help="Algorithm to use (must be 'det' or 'prob'). [%(default)s]")

    seeding_group = p.add_mutually_exclusive_group()
    seeding_group.add_argument(
        '--npv', dest='npv', action='store', metavar='NBR', type=int,
        help='Number of seeds per voxel. [1]')
    seeding_group.add_argument(
        '--nt', dest='nt', action='store', metavar='NBR', type=int,
        help='Total number of seeds. Replaces --npv and --ns.')
    seeding_group.add_argument(
        '--ns', dest='ns', action='store', metavar='NBR', type=int,
        help='Number of streamlines to estimate. Replaces --npv and\n' +
        '--nt. No multiprocessing is used.')

    p.add_argument(
        '--skip', dest='skip', action='store',
        metavar='NBR', type=int, default=0,
        help='Skip the first NBR generated seeds / NBR seeds per voxel\n' +
        '(--nt / --npv). Not working with --ns. [%(default)s]')
    p.add_argument(
        '--random', dest='random', action='store', metavar='RANDOM', type=int,
        default=0,
        help='Initial value for the random number generator. [%(default)s]')

    p.add_argument(
        '--step', dest='step_size', action='store',
        metavar='STEP', type=float, default=0.5,
        help='Step size in mm. [%(default)s]')

    # p.add_argument(
    #     '--rk_order', action='store', metavar='ORDER', type=int, default=2,
    #     choices=[1, 2, 4],
    #     help='The order of the Runge-Kutta integration used for \nthe step ' +
    #          'function. Must be 1, 2 or 4. [%(default)s]\nAs a rule of thumb' +
    #          ', doubling the rk_order will double \nthe computation time ' +
    #          'in the worst case.')

    # deviation_angle_group = p.add_mutually_exclusive_group()
    # deviation_angle_group.add_argument(
    #     '--theta', dest='theta', action='store',
    #     metavar='ANGLE', type=float,
    #     help="Maximum angle between 2 steps. ['det'=45, 'prob'=20]")
    # deviation_angle_group.add_argument(
    #     '--curvature', dest='curvature', action='store',
    #     metavar='RADIUS', type=float,
    #     help='Minimum radius of curvature R in mm. Replaces --theta.')
    p.add_argument(
        '--maxL_no_dir', dest='maxL_no_dir', action='store',
        metavar='MAX', type=float, default=1,
        help='Maximum length without valid direction, in mm. [%(default)s]')

    p.add_argument(
        '--sfthres', dest='sf_threshold', action='store',
        metavar='THRES', type=float, default=0.1,
        help='Spherical function relative threshold. [%(default)s]')
    p.add_argument(
        '--sfthres_init', dest='sf_threshold_init', action='store',
        metavar='THRES', type=float, default=0.5,
        help='Spherical function relative threshold value\nfor the initial ' +
        'direction. [%(default)s]')
    p.add_argument(
        '--minL', dest='min_length', action='store',
        metavar='MIN', type=float, default=10,
        help='Minimum length of a streamline in mm. [%(default)s]')
    p.add_argument(
        '--maxL', dest='max_length', action='store',
        metavar='MAX', type=int, default=300,
        help='Maximum length of a streamline in mm. [%(default)s]')

    p.add_argument(
        '--sh_interp', dest='field_interp', action='store',
        metavar='INTERP', type=str, default='tl', choices=['nn', 'tl'],
        help="Spherical harmonic interpolation: \n'nn' (nearest-neighbor) " +
        "or 'tl' (trilinear). [%(default)s]")
    p.add_argument(
        '--mask_interp', dest='mask_interp', action='store',
        metavar='INTERP', type=str, default='nn', choices=['nn', 'tl'],
        help="Mask interpolation:\n'nn' (nearest-neighbor) or 'tl' " +
        "(trilinear). [%(default)s]")

    p.add_argument(
        '--single_direction', dest='is_single_direction', action='store_true',
        help="If set, tracks in one direction only (forward or \nbackward) " +
             "given the initial seed. The direction is \n" +
             "randomly drawn from the ODF.")
    p.add_argument(
        '--processes', dest='nbr_processes', action='store', metavar='NBR',
        type=int, default=0,
        help='Number of sub processes to start. [cpu count]')
    p.add_argument(
        '--load_data', action='store_true', dest='isLoadData',
        help='If set, loads data in memory for all processes. \nIncreases ' +
             'the speed, and the memory requirements.')
    # p.add_argument(
    #     '--compress', action='store', dest='compress', type=float,
    #     help='If set, will compress streamlines. The parameter\nvalue is ' +
    #          'the distance threshold. A rule of thumb\nis to set it to ' +
    #          '0.1mm for deterministic\nstreamlines and ' +
    #          '0.2mm for probabilitic streamlines.')
    p.add_argument(
        '--save_seeds', action='store_true',
        help='If set, each streamline generated will save ' +
             'its 3D seed point in the TRK file using `seed` in ' +
             'the \'data_per_streamline\' attribute')
    p.add_argument(
        '-f', action='store_true', dest='isForce',
        help='If set, overwrites output file.')
    p.add_argument(
        '-v', action='store_true', dest='isVerbose',
        help='If set, produces verbose output.')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    param = {}

    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    output_filename = args.output_file

    if os.path.isfile(output_filename):
        if args.isForce:
            logging.debug('Overwriting "{0}".'.format(output_filename))
        else:
            parser.error('"{0}" already exists! Use -f to overwrite it.'
                         .format(output_filename))

    if not np.any([args.nt, args.npv, args.ns]):
        args.npv = 1

    if not args.min_length > 0:
        parser.error('minL must be > 0, {0}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={0}mm, maxL={1}mm).'
                     .format(args.min_length, args.max_length))

    # if args.theta is not None:
    #     theta = gm.math.radians(args.theta)
    # elif args.curvature > 0:
    #     theta = get_max_angle_from_curvature(args.curvature, args.step_size)
    # elif args.algo == 'prob':
    #     theta = gm.math.radians(20)
    # else:
    #     theta = gm.math.radians(45)

    if args.mask_interp == 'nn':
        mask_interpolation = 'nearest'
    elif args.mask_interp == 'tl':
        mask_interpolation = 'trilinear'
    else:
        parser.error("--mask_interp has wrong value. See the help (-h).")
        return

    if args.field_interp == 'nn':
        field_interpolation = 'nearest'
    elif args.field_interp == 'tl':
        field_interpolation = 'trilinear'
    else:
        parser.error("--sh_interp has wrong value. See the help (-h).")
        return

    param['random'] = args.random
    param['skip'] = args.skip
    param['mask_interp'] = mask_interpolation
    param['field_interp'] = field_interpolation
    param['sf_threshold'] = args.sf_threshold
    param['sf_threshold_init'] = args.sf_threshold_init
    param['step_size'] = args.step_size
    param['max_length'] = args.max_length
    param['min_length'] = args.min_length
    param['max_nbr_pts'] = int(param['max_length'] / param['step_size'])
    param['min_nbr_pts'] = int(param['min_length'] / param['step_size']) + 1
    param['is_single_direction'] = args.is_single_direction
    param['nbr_seeds'] = args.nt if args.nt is not None else 0
    param['nbr_seeds_voxel'] = args.npv if args.npv is not None else 0
    param['nbr_streamlines'] = args.ns if args.ns is not None else 0
    param['max_no_dir'] = int(math.ceil(args.maxL_no_dir / param['step_size']))
    param['is_all'] = False
    param['is_keep_single_pts'] = False
    # r+ is necessary for interpolation function in cython who
    # need read/write right
    param['mmap_mode'] = None if args.isLoadData else 'r+'
    logging.debug('Tractography parameters:\n{0}'.format(param))

    seed_img = nib.load(args.seed_file)
    seed = Seed(seed_img)
    if args.npv:
        param['nbr_seeds'] = len(seed.seeds) * param['nbr_seeds_voxel']
        param['skip'] = len(seed.seeds) * param['skip']
    if len(seed.seeds) == 0:
        parser.error('"{0}" does not have voxels value > 0.'
                     .format(args.seed_file))

    mask = Dataset(nib.load(args.atlas), param['mask_interp'])

    dataset = Dataset(nib.load(args.sh_file), param['field_interp'])

    import json
    with open(args.config) as json_file:
        config = json.load(json_file)
    tracker = probabilisticTracker(mask, dataset, args.sh_basis, config, param)

    start = time.time()
    # if args.compress:
    #     if args.compress < 0.001 or args.compress > 1:
    #         logging.warn('You are using an error rate of {}.\n'
    #                      .format(args.compress) +
    #                      'We recommend setting it between 0.001 and 1.\n' +
    #                      '0.001 will do almost nothing to the tracts while ' +
    #                      '1 will higly compress/linearize the tracts')

    #     streamlines, seeds = track(tracker, mask, seed, param, compress=True,
    #                                compression_error_threshold=args.compress,
    #                                nbr_processes=args.nbr_processes,
    #                                pft_tracker=None,
    #                                save_seeds=args.save_seeds)
    
    streamlines, seeds = track(tracker, mask, seed, param,
                                nbr_processes=args.nbr_processes,
                                save_seeds=args.save_seeds)
    print("lol")
    sft = StatefulTractogram(streamlines, args.seed_file, Space.VOXMM)
    save_tractogram(sft, args.output_file, bbox_valid_check=False)
    
    str_time = "%.2f" % (time.time() - start)
    print(str(len(streamlines)) + " streamlines, done in " +
                  str_time + " seconds.")

if __name__ == "__main__":
    main()
