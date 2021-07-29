#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Return the number of streamlines in a tractogram. Only support trk and tck in
order to support the lazy loading from nibabel.
"""

import argparse
import os
import seaborn as sn
import pandas
import numpy as np
import matplotlib

from scilpy.io.streamlines import lazy_streamlines_count
from scilpy.io.utils import add_json_args


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', nargs="+",
                   help='Path of the input tractogram file.')
    add_json_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    stats = {}
    for i in args.in_tractogram:
        id = os.path.basename(os.path.dirname(os.path.dirname(i)))
        bundle_name, _ = os.path.splitext(os.path.basename(i))
        bundle_name = bundle_name[len(id+'__'):]
        bundle_name = bundle_name[0:len(bundle_name) - len("_cleaned")]
        if bundle_name in stats:
            stats[bundle_name][id] = int(lazy_streamlines_count(i))
        else:
            stats[bundle_name] = {id: int(lazy_streamlines_count(i))}

    test = pandas.DataFrame.from_dict(stats)
    ax = sn.boxplot(data=test)
    ax = sn.swarmplot(data=test, color=".25")
    ax.set
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    for i in test.keys():
        val = test[i].dropna().values
        mean = np.mean(val)
        std = np.std(val)
        for index, value in test[i].iteritems():
            if value <= mean + std or value >= mean - std:
                print(index, i, "Pass")
            elif value <= mean + 2 * std or value >= mean -2 * std:
                print(index, i, "Warning")
            else:
                print(index, i, "Fail")
    ax.figure.set_size_inches(18.5, 8.5)
    matplotlib.pyplot.savefig("boxplot.png", dpi=100, orientation='landscape')

if __name__ == '__main__':
    main()
