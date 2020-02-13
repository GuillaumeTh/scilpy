#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute include and exclude maps, and the seeding interface mask from partial
volume estimation (PVE) maps. Maps should have values in [0,1], gm+wm+csf=1 in
all voxels of the brain, gm+wm+csf=0 elsewhere.

References: Girard, G., Whittingstall K., Deriche, R., and Descoteaux, M.
(2014). Towards quantitative connectivity analysis: reducing tractography
biases. Neuroimage.
"""

from __future__ import division

import argparse
import logging

import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'segmentation',
        help='Tissue segmentation from dmriseg.')

    p.add_argument(
        '--include', metavar='filename', default='map_include.nii.gz',
        help='Output include map (nifti). [map_include.nii.gz]')
    p.add_argument(
        '--exclude', metavar='filename', default='map_exclude.nii.gz',
        help='Output exclude map (nifti). [map_exclude.nii.gz]')
    p.add_argument(
        '--seeding_mask', metavar='filename', default='seeding_mask.nii.gz',
        help='Output seeding mask (nifti). [seeding_mask.nii.gz]')
    p.add_argument(
        '-t', dest='int_thres', metavar='THRESHOLD', type=float, default=0.1,
        help='Minimum wm PVE values in a voxel to be in to the '
             'seeding_mask. [0.1]')
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    assert_inputs_exist(parser, [args.segmentation])
    assert_outputs_exist(parser, args,
                          [args.include, args.exclude, args.seeding_mask])

    min_value = 0.0001

    segmentation_img = nib.load(args.segmentation)
    segmentation = segmentation_img.get_data()
    background = segmentation[:,:,:,0]
    wm_pve = segmentation[:,:,:,1]
    gm_pve = segmentation[:,:,:,2]
    ventricules_pve = segmentation[:,:,:,3]
    putamen_pve = segmentation[:,:,:,4]
    pallidum_pve = segmentation[:,:,:,5]
    hippocampus_pve = segmentation[:,:,:,6]
    caudate_pve = segmentation[:,:,:,7]
    amygdala_pve = segmentation[:,:,:,8]
    accumbens_pve = segmentation[:,:,:,9]
    plexus_pve = segmentation[:,:,:,10]
    thalamus_pve = segmentation[:,:,:,11]

    include_map = gm_pve
    include_map[background > 0.5] = 1

    exclude_map = ventricules_pve
    exclude_map += putamen_pve
    exclude_map += pallidum_pve
    exclude_map += hippocampus_pve
    exclude_map += caudate_pve
    exclude_map += amygdala_pve
    exclude_map += accumbens_pve
    exclude_map += plexus_pve
    exclude_map += thalamus_pve
    exclude_map[exclude_map < 0.5] = 0

    labels = np.argmax(segmentation, axis=3)
    seeding_mask = np.zeros(wm_pve.shape)
    seeding_mask[labels == 1] = 1

    nib.Nifti1Image(include_map.astype('float32'),
                    segmentation_img.affine).to_filename(args.include)
    nib.Nifti1Image(exclude_map.astype('float32'),
                    segmentation_img.affine).to_filename(args.exclude)
    nib.Nifti1Image(seeding_mask.astype('float32'),
                    segmentation_img.affine).to_filename(args.seeding_mask)


if __name__ == "__main__":
    main()
