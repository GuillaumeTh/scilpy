#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptative tractography
"""
import argparse
import ast
import logging
import math
import time

import dipy.core.geometry as gm
import json
import nibabel as nib

from dipy.io.stateful_tractogram import StatefulTractogram, Space, set_sft_logger_level
from dipy.io.stateful_tractogram import Origin
from dipy.io.streamline import save_tractogram

from scilpy.io.utils import (
    add_processes_arg,
    add_sphere_arg,
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    verify_compression_th,
)
from scilpy.image.datasets import DataVolume
from scilpy.tracking.propagator import DynamicODFPropagator, ODFPropagator
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.tools import get_theta
from scilpy.tracking.tracker import TissueTracker
from scilpy.tracking.utils import (
    add_mandatory_options_tracking,
    add_out_options,
    add_seeding_options,
    add_sh_basis_args,
    verify_streamline_length_options,
    verify_seed_options,
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    p.add_argument(
        "in_odf",
        help="File containing the orientation diffusion function \n"
        "as spherical harmonics file (.nii.gz). Ex: ODF or "
        "fODF.",
    )
    p.add_argument("in_seed", help="Seeding mask (.nii.gz).")
    p.add_argument(
        "in_atlas",
        help="Atlas image (.nii.gz).",
    )
    p.add_argument("config", help="Config file")
    p.add_argument(
        "out_tractogram", help="Tractogram output file (must be .trk or .tck)."
    )

    track_g = p.add_argument_group("Tracking options")
    track_g.add_argument(
        "--min_length",
        type=float,
        default=10.0,
        metavar="m",
        help="Minimum length of a streamline in mm. " "[%(default)s]",
    )
    track_g.add_argument(
        "--max_length",
        type=float,
        default=300.0,
        metavar="M",
        help="Maximum length of a streamline in mm. " "[%(default)s]",
    )
    track_g.add_argument(
        "--sfthres",
        dest="sf_threshold",
        metavar="sf_th",
        type=float,
        default=0.1,
        help="Spherical function relative threshold. " "[%(default)s]",
    )
    add_sh_basis_args(track_g)

    add_sphere_arg(track_g, symmetric_only=False)
    track_g.add_argument(
        "--sfthres_init",
        metavar="sf_th",
        type=float,
        default=0.5,
        dest="sf_threshold_init",
        help="Spherical function relative threshold value "
        "for the \ninitial direction. [%(default)s]",
    )
    track_g.add_argument(
        "--max_invalid_length",
        metavar="MAX",
        type=float,
        default=1,
        help="Maximum length without valid direction, in mm. " "[%(default)s]",
    )
    track_g.add_argument(
        "--forward_only",
        action="store_true",
        help="If set, tracks in one direction only (forward) "
        "given the \ninitial seed. The direction is "
        "randomly drawn from the ODF.",
    )
    track_g.add_argument(
        "--sh_interp",
        default="trilinear",
        choices=["nearest", "trilinear"],
        help="Spherical harmonic interpolation: "
        "nearest-neighbor \nor trilinear. [%(default)s]",
    )
    track_g.add_argument(
        "--mask_interp",
        default="nearest",
        choices=["nearest", "trilinear"],
        help="Mask interpolation: nearest-neighbor or " "trilinear. [%(default)s]",
    )
    track_g.add_argument(
        "--percentage_stop",
        default=0.25,
        help="Percentage of seeds that can stop in nuclei [%(default)s]",
    )

    add_seeding_options(p)

    r_g = p.add_argument_group("Random seeding options")
    r_g.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="Initial value for the random number generator. " "[%(default)s]",
    )
    r_g.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N random number. \n"
        "Useful if you want to create new streamlines to "
        "add to \na previously created tractogram with a "
        "fixed --rng_seed.\nEx: If tractogram_1 was created "
        "with -nt 1,000,000, \nyou can create tractogram_2 "
        "with \n--skip 1,000,000.",
    )

    m_g = p.add_argument_group("Memory options")
    add_processes_arg(m_g)
    m_g.add_argument(
        "--set_mmap_to_none",
        action="store_true",
        help="If true, use mmap_mode=None. Else mmap_mode='r+'. "
        "\nUsed in np.load(data_file_info). TO BE CLEANED",
    )

    add_out_options(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error(
            "Invalid output streamline file format (must be trk or "
            + "tck): {0}".format(args.out_tractogram)
        )

    inputs = [args.in_odf, args.in_seed, args.in_mask]
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    # TODO adapt with the wm step size in the config file
    # Here we hard coded the step size to 0.5
    max_nbr_pts = int(args.max_length / 0.5)
    min_nbr_pts = int(args.min_length / 0.5) + 1
    max_invalid_dirs = int(math.ceil(args.max_invalid_length / 0.5))

    # r+ is necessary for interpolation function in cython who need read/write
    # rights
    mmap_mode = None if args.set_mmap_to_none else "r+"

    logging.debug("Loading seeding mask.")
    seed_img = nib.load(args.in_seed)
    seed_data = seed_img.get_fdata(caching="unchanged", dtype=float)
    seed_res = seed_img.header.get_zooms()[:3]
    seed_generator = SeedGenerator(seed_data, seed_res, args.percentage_stop)
    if args.npv:
        # toDo. This will not really produce n seeds per voxel, only true
        #  in average.
        nbr_seeds = len(seed_generator.seeds) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds)
    if len(seed_generator.seeds) == 0:
        parser.error(
            'Seed mask "{}" does not have any voxel with value > 0.'.format(
                args.in_seed
            )
        )

    logging.debug("Loading tracking mask.")
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching="unchanged", dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, args.mask_interp)

    logging.debug("Loading ODF SH data.")
    odf_sh_img = nib.load(args.in_odf)
    odf_sh_data = odf_sh_img.get_fdata(caching="unchanged", dtype=float)
    odf_sh_res = odf_sh_img.header.get_zooms()[:3]
    dataset = DataVolume(odf_sh_data, odf_sh_res, args.sh_interp)

    logging.debug("Loading Configuration File.")
    with open(args.config) as config_file:
        config = ast.literal_eval(config_file.read())

    logging.debug("Instantiating propagator.")
    # propagator = ODFPropagator(
    #     dataset, args.step_size, args.rk_order, args.algo, args.sh_basis,
    #     args.sf_threshold, args.sf_threshold_init, theta, args.sphere)
    propagator = DynamicODFPropagator(
        dataset, args.sh_basis, args.sf_threshold, args.sf_threshold_init, args.sphere
    )

    tracker = TissueTracker(
        propagator,
        mask,
        seed_generator,
        config,
        nbr_seeds,
        min_nbr_pts,
        max_nbr_pts,
        max_invalid_dirs,
        args.compress,
        args.nbr_processes,
        args.save_seeds,
        mmap_mode,
        args.rng_seed,
        args.forward_only,
        args.skip,
    )

    start = time.time()
    logging.debug("Tracking...")
    streamlines, seeds = tracker.track()

    str_time = "%.2f" % (time.time() - start)
    logging.debug(
        "Tracked {} streamlines (out of {} seeds), in {} seconds.\n"
        "Now saving...".format(len(streamlines), nbr_seeds, str_time)
    )

    # save seeds if args.save_seeds is given
    data_per_streamline = {"seeds": seeds} if args.save_seeds else {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level("WARNING")

    # Compared with scil_compute_local_tracking, using sft rather than
    # LazyTractogram to deal with space.
    # Contrary to scilpy or dipy, where space after tracking is vox, here
    # space after tracking is voxmm.
    # Smallest possible streamline coordinate is (0,0,0), equivalent of
    # corner origin (TrackVis)
    sft = StatefulTractogram(
        streamlines,
        mask_img,
        Space.VOXMM,
        Origin.TRACKVIS,
        data_per_streamline=data_per_streamline,
    )
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
