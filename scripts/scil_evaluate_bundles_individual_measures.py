#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate basic measurements of bundles, all at once.
All tractograms must be in the same space (aligned to one reference)
The computed measures are:
volume, volume_endpoints, streamlines_count, avg_length, std_length,
min_length, max_length, span, curl, diameter, elongation, mean_curvature
Note: volume is calculated differently in scilpy with density
"""

import argparse
import json
import logging
import multiprocessing

from dipy.io.streamline import load_tractogram
from dipy.tracking.metrics import mean_curvature
from dipy.tracking.utils import length
import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             link_bundles_and_reference,
                             validate_nbr_processes)

from scilpy.tractanalysis.reproducibility_measures \
    import get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.streamlines import uniformize_bundle_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='Path of the input bundles.')
    p.add_argument('out_json',
                   help='Path of the output file.')
    p.add_argument('--group_statistics', action='store_true',
                   help='Show average measures \n'
                        '[%(default)s].')
    add_reference_arg(p)
    add_processes_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p


def compute_measures(filename_tuple):
    sft = load_tractogram(filename_tuple[0], filename_tuple[1])
    _, dimensions, voxel_size, _ = sft.space_attributes
    uniformize_bundle_sft(sft)
    nbr_streamlines = len(sft)
    if not nbr_streamlines:
        logging.warning('{} is empty'.format(filename_tuple[0]))
        return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                         'avg_length', 'std_length', 'min_length',
                         'max_length', 'span', 'curl', 'diameter',
                         'elongation', 'mean_curvature'],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    length_list = list(length(list(sft.streamlines)))
    length_avg = float(np.average(length_list))
    length_std = float(np.std(length_list))
    length_min = float(np.min(length_list))
    length_max = float(np.max(length_list))

    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines
    density = compute_tract_counts_map(streamlines, dimensions)
    endpoints_density = get_endpoints_density_map(streamlines, dimensions)

    span_list = list(map(compute_span, list(sft.streamlines)))
    span = float(np.average(span_list))
    curl = length_avg / span
    # volume is calculated differently in scilpy with density
    volume = np.count_nonzero(density) * np.product(voxel_size)
    diameter = 2 * np.sqrt(volume / (np.pi * length_avg))
    elon = length_avg / diameter

    curvature_list = np.zeros((nbr_streamlines,))
    for i in range(nbr_streamlines):
        curvature_list[i] = mean_curvature(sft.streamlines[i])

    return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                     'avg_length', 'std_length', 'min_length', 'max_length',
                     'span', 'curl', 'diameter', 'elongation',
                     'mean_curvature'],
                    [volume, np.count_nonzero(endpoints_density) *
                     np.product(voxel_size), nbr_streamlines,
                     length_avg, length_std, length_min, length_max,
                     span, curl, diameter, elon,
                     float(np.mean(curvature_list))]))


def compute_span(streamline_cords):
    xyz = np.asarray(streamline_cords)
    if xyz.shape[0] < 2:
        return 0
    dists = np.sqrt((np.diff([xyz[0], xyz[-1]], axis=0) ** 2).sum(axis=1))
    return np.sum(dists)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_bundles)
    assert_outputs_exist(parser, args, args.out_json)

    nbr_cpu = validate_nbr_processes(parser, args)
    bundles_references_tuple_extended = link_bundles_and_reference(
        parser, args, args.in_bundles)

    if nbr_cpu == 1:
        all_measures_dict = []
        for i in bundles_references_tuple_extended:
            all_measures_dict.append(compute_measures(i))
    else:
        pool = multiprocessing.Pool(nbr_cpu)
        all_measures_dict = pool.map(compute_measures,
                                     bundles_references_tuple_extended)
        pool.close()
        pool.join()

    output_measures_dict = {}
    for measure_dict in all_measures_dict:
        # Empty bundle should not make the script crash
        if measure_dict is not None:
            for measure_name in measure_dict.keys():
                # Create an empty list first
                if measure_name not in output_measures_dict:
                    output_measures_dict[measure_name] = []
                output_measures_dict[measure_name].append(
                    measure_dict[measure_name])
    # add set stats if there are more than one bundle
    if args.group_statistics:
        set_total_length = np.sum(
            np.multiply(output_measures_dict['avg_length'],
                        output_measures_dict['streamlines_count']))
        set_total_span = np.sum(
            np.multiply(output_measures_dict['span'],
                        output_measures_dict['streamlines_count']))
        set_total_volume = np.sum(
            np.multiply(output_measures_dict['volume'],
                        output_measures_dict['streamlines_count']))
        set_streamlines_count = \
            np.sum(output_measures_dict['streamlines_count'])
        output_measures_dict['set_stats'] = {}
        output_measures_dict['set_stats']['set_streamlines_count'] = \
            float(set_streamlines_count)
        output_measures_dict['set_stats']['set_avg_length'] = \
            set_total_length / set_streamlines_count
        output_measures_dict['set_stats']['set_max_length'] = \
            float(np.min(output_measures_dict['max_length']))
        output_measures_dict['set_stats']['set_min_length'] = \
            float(np.min(output_measures_dict['min_length']))
        output_measures_dict['set_stats']['set_avg_span'] = \
            set_total_span / set_streamlines_count
        output_measures_dict['set_stats']['set_avg_volume'] = \
            set_total_volume / set_streamlines_count
    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
