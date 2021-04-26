#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute brain extraction from ANTs
"""

import argparse

from ants import image_read, image_write
from antspynet.utilities import brain_extraction


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('image',
                   help='Path of input image.')
    p.add_argument('mask',
                   help='Brain extraction mask.')

    p.add_argument('--type', choices=["t1", "b0"],
                   help='Brain extraction mask.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    img = image_read(args.image)

    type_img = args.type
    if args.type == "b0":
        type_img = "t2"

    probability = brain_extraction(img, type_img)

    image_write(probability, args.mask)


if __name__ == "__main__":
    main()
