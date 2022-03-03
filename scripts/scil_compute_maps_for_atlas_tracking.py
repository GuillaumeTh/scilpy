#!/usr/bin/env python3
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

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'atlas',
        help='White matter PVE map (nifti). From normal FAST output, has a '
             'PVE_2 name suffix.')

   
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    atlas = nib.load(args.atlas)
    atlas_d = atlas.get_data()
    atlas_d[atlas_d == 3] = 12
    for i in [4, 5, 6, 7, 8, 9, 11]:
        atlas_d[atlas_d == i] = 3
    atlas_d[atlas_d == 12] = 4
    atlas_d[atlas_d == 10] = 4

    nib.Nifti1Image(atlas_d.astype('int8'),
                    atlas.affine).to_filename("atlas_p.nii.gz")

if __name__ == "__main__":
    main()
