# -*- coding: utf-8 -*-

import random

import dipy.core.geometry as gm
import numpy as np

from scilpy.tracking.trackingField import (SphericalHarmonicField,
                                           TrackingDirection)


class tissue_tracker(object):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold,
                 sf_threshold_init, config, dipy_sphere):
        self.tracker = tracker
        self.tissue_value = self.get_tissue_value()
        theta = gm.math.radians(config[self.tissue_value]['theta'])
        self.tracking_field = SphericalHarmonicField(odf_dataset, basis,
                                                     sf_threshold,
                                                     sf_threshold_init, theta,
                                                     dipy_sphere)
        self.step_size = config[self.tissue_value]['step_size']

    def get_segment(self, pos, v_in):
        #Default RK1 else new definition in tissues.
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        newDir = self.getValidDirection(
            pos + 0.5 * self.step_size * np.array(dir1), dir1)[1]
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction

    def get_tissue_value(self):
        pass
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        #Default append and stop tracking
        line.append(new_pos)
        line_dirs.append(new_dir)
        return line, line_dirs, True
    
    def getValidDirection(self, pos, v_in):
        is_valid_direction = True
        v_out = self.tracker.get_direction(pos, v_in)
        if v_out is None:
            is_valid_direction = False
            v_out = v_in

        return is_valid_direction, v_out


class wm_tissue_tracker(tissue_tracker):

    def get_tissue_value(self):
        return "1"

    def get_segment(self, pos, v_in):
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        v1 = np.array(dir1)
        dir2 = self.getValidDirection(pos + 0.5 * self.step_size * v1, dir1)[1]
        v2 = np.array(dir2)
        dir3 = self.getValidDirection(pos + 0.5 * self.step_size * v2, dir2)[1]
        v3 = np.array(dir3)
        dir4 = self.getValidDirection(pos + self.step_size * v3, dir3)[1]
        v4 = np.array(dir4)

        newV = (v1 + 2 * v2 + 2 * v3 + v4) / 6
        newDir = TrackingDirection(newV, dir1.index)
        newPos = pos + self.step_size * newV

        return newPos, newDir, is_valid_direction
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        line.append(new_pos)
        line_dirs.append(new_dir)
        return line, line_dirs, False


class gm_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "2"

    def get_segment(self, pos, v_in):
        return None, None, False
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        line.append(new_pos)
        line_dirs.append(new_dir)
        return line, line_dirs, True


class csf_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "3"

    def get_segment(self, pos, v_in):
        return None, None, False
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        return None, None, True


class background_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "0"

    def get_segment(self, pos, v_in):
        return None, None, False

    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        return None, None, True


class putamen_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "4"

    def get_segment(self, pos, v_in):
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        newDir = self.getValidDirection(
            pos + 0.5 * self.step_size * np.array(dir1), dir1)[1]
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction

    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        max_iter_in_nulcei = 20
        cur_nb_iter = np.random.randint(1, max_iter_in_nulcei)
        last_n_steps = line[-cur_nb_iter - 1:-1]
        history = []
        is_stopping = False
        if len(line) > max_iter_in_nulcei:
            for i in last_n_steps:
                history.append(self.tracker.atlas.getPositionValueFromAtlas(i[0], i[1], i[2]))
            is_stopping = len(np.unique(history)) == 1 and np.unique(history)[0] == int(self.tissue_value)
        if is_stopping:
            #Stop and new line
            line.append(new_pos)
            line_dirs.append(new_dir)
            return line, line_dirs, True
        else:
            #Continue line
            line.append(new_pos)
            line_dirs.append(new_dir)
            return line, line_dirs, False
            # new_line = [new_pos]
            # new_dirs = [TrackingDirection(-np.array(new_dir), new_dir.index)]
            # history = []
            # while len(new_line) < self.tracker.max_nbr_points:
            #     new_pos, new_dir, valid_direction_nbr = self.tracker.propagate(
            #         new_line[-1], new_dirs[-1])
            #
            #     if new_pos is None or new_dir is None:
            #         return None, None, True
            #     history.append(self.tracker.get_tissue_tracker_function(new_pos).tissue_value)
            #     is_finished = False
            #     if len(new_line) == 1:
            #         new_line.append(new_pos)
            #         new_dirs.append(new_dir)
            #     elif len(np.where(history[-5:-1] == self.tissue_value)) > 5:
            #         return None, None, True
            #     elif self.tracker.get_tissue_tracker_function(new_line[-1]).tissue_value == self.tracker.get_tissue_tracker_function(new_line[-2]).tissue_value:
            #         new_line.append(new_pos)
            #         new_dirs.append(new_dir)
            #     else:
            #         new_line, new_dirs, is_finished = append_to_line(self.tracker, valid_direction_nbr, new_pos, new_dir, new_line, new_dirs)
            #
            #     if new_line is None or new_dirs is None:
            #         return None, None, True
            #
            #     if is_finished:
            #         return [line, new_line], [line_dirs, new_dirs], True


class pallidum_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "5"

    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        max_iter_in_nulcei = 20
        cur_nb_iter = np.random.randint(1, max_iter_in_nulcei)
        last_n_steps = line[-cur_nb_iter - 1:-1]
        history = []
        is_stopping = False
        if len(line) > max_iter_in_nulcei:
            for i in last_n_steps:
                history.append(self.tracker.atlas.getPositionValueFromAtlas(i[0], i[1], i[2]))
            is_stopping = len(np.unique(history)) == 1 and np.unique(history)[0] == int(self.tissue_value)
        if is_stopping:
            #Stop and new line
            line.append(new_pos)
            line_dirs.append(new_dir)
            return line, line_dirs, True
        else:
            #Continue line
            line.append(new_pos)
            line_dirs.append(new_dir)
            return line, line_dirs, False


class hippocampus_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "6"


class caudate_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "7"


class amygdala_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "8"


class accumbens_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "9"


class thalamus_tissue_tracker(tissue_tracker):
    def get_tissue_value(self):
        return "11"


def append_to_line(tracker, tissue_key, new_pos, new_dir, line, line_dirs):
    if tissue_key in tissue_trackers.keys():
        tracker_f = tracker.tracker_functions[tissue_key]
        return tracker_f._append_to_line(new_pos, new_dir, line, line_dirs)
    else:
        return None, None, True


tissue_trackers = {
    "0": background_tissue_tracker,
    "1": wm_tissue_tracker,
    "2": gm_tissue_tracker,
    "3": csf_tissue_tracker,
    "4": putamen_tissue_tracker,
    "5": pallidum_tissue_tracker,
    "6": hippocampus_tissue_tracker,
    "7": caudate_tissue_tracker,
    "8": amygdala_tissue_tracker,
    "9": accumbens_tissue_tracker,
    "11": thalamus_tissue_tracker,
    "12": background_tissue_tracker,
    "10": background_tissue_tracker
}

# tissue_trackers = {
#     "0": background_tissue_tracker,
#     "1": wm_tissue_tracker,
#     "2": gm_tissue_tracker,
#     "3": csf_tissue_tracker,
#     "4": nuclei_tissue_tracker,
#     "5": nuclei_tissue_tracker,
#     "6": nuclei_tissue_tracker,
#     "7": nuclei_tissue_tracker,
#     "8": nuclei_tissue_tracker,
#     "9": nuclei_tissue_tracker,
#     "11": nuclei_tissue_tracker,
#     "12": background_tissue_tracker,
#     "10": background_tissue_tracker
# }

