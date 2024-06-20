#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Add information to each streamline from a file. Can be for example
SIFT2 weights, processing information, bundle IDs, etc.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck).')
    p.add_argument('dps_file',
                   help='File containing the data to add to streamlines')
    p.add_argument('dps_key',
                   help='Where to store the data in the tractogram.')
    p.add_argument('out_tractogram',
                   help='Output tractogram (.trk or .tck).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_tractogram, args.dps_file],
                        args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    data = np.squeeze(load_matrix_in_any_format(args.dps_file))

    if len(sft) != data.shape[0]:
        raise ValueError('Data must have as many entries ({}) as there are'
                         ' streamlines ({}).'.format(data.shape[0], len(sft)))

    sft.data_per_streamline[args.dps_key] = data
    # Saving
    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
