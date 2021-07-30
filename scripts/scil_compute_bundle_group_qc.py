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
import tqdm
import json

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
    for i in tqdm.tqdm(args.in_tractogram):
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
    ax = sn.stripplot(data=test, color=".25")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    qc_report = []
    settings = {"type": "settings", "data": [], "username": "auto_qc", "date": ""}
    report = {"type": "report", "data": []}
    report_dict = {}
    subj = set([])
    for i in test.keys():
        val = test[i].dropna().values
        mean = np.mean(val)
        std = np.std(val)
        print(mean, std, val)
        for index, value in test[i].dropna().iteritems():
            if value == np.nan:
                continue
            if value < 50:
                rating = "Fail"
                comment = "Under 50 streamlines"
            elif value <= mean + std and value >= mean - std:
                rating = "Pass"
                comment = ""
            elif value <= (mean + 2 * std) and value >= (mean - 2 * std):
                rating = "Warning"
                comment = "Number of streamlines between mean +- 2 std"
            else:
                rating = "Fail"
                comment = "Number of streamlines under or upper than mean +- 1.5 std"
            if i in report_dict:
                report_dict[i][index] = {"status": rating, "comments": comment}
            else:
                report_dict[i] = {index: {"status": rating, "comments": comment}}
            subj.add(index)

    b_name_rating = set([])
    for i in test.keys():
        if "_L" in i or "_R" in i:
            b_name = i.replace("_L", "").replace("_R", "")
            b_name_rating.add(b_name)

    for b_name in b_name_rating:
        for sub in subj:
            if b_name not in report_dict:
                report_dict[b_name] = {}
            if sub in report_dict[b_name+"_L"] and sub in report_dict[b_name+"_R"]:
                left = report_dict[b_name+"_L"][sub]["status"]
                right = report_dict[b_name+"_R"][sub]["status"]

                if left == "Pass" and right == "Pass":
                    report_dict[b_name][sub] = report_dict[b_name+"_L"][sub]

                if left == "Warning" and right == "Pass":
                    report_dict[b_name][sub] = report_dict[b_name+"_L"][sub]

                if left == "Pass" and right == "Warning":
                    report_dict[b_name][sub] = report_dict[b_name+"_R"][sub]

                if left == "Warning" and right == "Warning":
                    report_dict[b_name][sub] = report_dict[b_name+"_R"][sub]

                if left == "Fail" or right == "Fail":
                    if left == "Fail":
                        report_dict[b_name][sub] = report_dict[b_name+"_L"][sub]
                    else:
                        report_dict[b_name][sub] = report_dict[b_name+"_R"][sub]
            elif sub in report_dict[b_name+"_L"]:
                report_dict[b_name][sub] = {"status": "Fail", "comments": "Missing left bundle"}
            elif sub in report_dict[b_name+"_R"]:
                report_dict[b_name][sub] = {"status": "Fail", "comments": "Missing right bundle"}

        report_dict.pop(b_name + "_L")
        report_dict.pop(b_name + "_R")

    for i in report_dict:
        for index in report_dict[i]:
            rating = report_dict[i][index]["status"]
            comment = report_dict[i][index]["comments"]
            report["data"].append({"qc": "imgs", "status": rating, "comments": comment, "filename": index+"__"+i+"_cleaned.png"})
    qc_report.append(settings)
    qc_report.append(report)
    with open('report.json', 'w') as f:
        json.dump(qc_report, f)
    ax.figure.set_size_inches(18.5, 8.5)
    matplotlib.pyplot.savefig("boxplot.png", dpi=100, orientation='landscape')


if __name__ == '__main__':
    main()
