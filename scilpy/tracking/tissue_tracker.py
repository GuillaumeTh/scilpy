# -*- coding: utf-8 -*-

from scilpy.tracking.trackingField import SphericalHarmonicField
from scilpy.tracking.trackingField import TrackingDirection
import numpy as np
import dipy.core.geometry as gm

class tissue_tracker(object):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tracker = tracker
        theta = gm.math.radians(config[self.tissue_value]['theta'])
        self.tracking_field = SphericalHarmonicField(odf_dataset, basis, sf_threshold, sf_threshold_init, theta, dipy_sphere)
        self.step_size = config[self.tissue_value]['step_size']
    
    def get_segment(self, pos, v_in):
        pass
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs):
        pass
    
    def getValidDirection(self, pos, v_in):
        is_valid_direction = True
        v_out = self.tracker.get_direction(pos, v_in)
        if v_out is None:
            is_valid_direction = False
            v_out = v_in

        return is_valid_direction, v_out
    
class wm_tissue_tracker(tissue_tracker):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tissue_value = "1"
        self.stop_in_tissue = False
        super(wm_tissue_tracker, self).__init__(tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere)

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
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs, stop_in_tissue=False):
        line.append(new_pos)
        line_dirs.append(new_dir)
        return line, line_dirs, False

class gm_tissue_tracker(tissue_tracker):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tissue_value = "2"
        self.stop_in_tissue = False
        super(gm_tissue_tracker, self).__init__(tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere)

    def get_segment(self, pos, v_in):
        return None, None, True
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs, stop_in_tissue=False):
        line.append(new_pos)
        line_dirs.append(new_dir)
        return line, line_dirs, True

class nuclei_tissue_tracker(tissue_tracker):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tissue_value = "3"
        self.config = config
        self.stop_in_tissue = True
        self.max_random_ending = config[self.tissue_value]['max_random_ending']
        self.min_distance = config[self.tissue_value]['min_distance_before_stop']
        super(nuclei_tissue_tracker, self).__init__(tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere)

    def get_segment(self, pos, v_in):
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        newDir = self.getValidDirection(
            pos + 0.5 * self.step_size * np.array(dir1), dir1)[1]
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs, stop_in_tissue=True):
        line.append(new_pos)
        line_dirs.append(new_dir)
        if stop_in_tissue:
            number = np.random.rand()
            if number >= self.max_random_ending:
                new_line = [new_pos]
                new_dirs = [new_dir]
                distance = 0
                while len(new_line) < self.tracker.max_nbr_points:
                    new_pos, new_dir, valid_direction_nbr = self.tracker.propagate(
                        new_line[-1], new_dirs[-1])
                    if valid_direction_nbr != self.tissue_value and valid_direction_nbr is not None:
                        distance += self.tracker.tracker_functions[valid_direction_nbr].step_size

                    if distance > self.min_distance:
                        stop_in_tissue=True
                    else:
                        stop_in_tissue=False

                    new_line, new_dirs, is_finished = append_to_line(self.tracker, valid_direction_nbr, new_pos, new_dir, new_line, new_dirs, stop_in_tissue=stop_in_tissue)

                    if new_line is None or new_dirs is None:
                        break

                    if is_finished:
                        return [line, new_line], [line_dirs, new_dirs], True
                return line, line_dirs, True
        return line, line_dirs, False
    
class csf_tissue_tracker(tissue_tracker):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tissue_value = "4"
        self.stop_in_tissue = False
        super(csf_tissue_tracker, self).__init__(tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere)

    def get_segment(self, pos, v_in):
        return None, None, None
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs, stop_in_tissue=False):
        return None, None, True

class background_tissue_tracker(tissue_tracker):
    def __init__(self, tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere):
        self.tissue_value = "0"
        self.stop_in_tissue = False
        super(background_tissue_tracker, self).__init__(tracker, odf_dataset, basis, sf_threshold, sf_threshold_init, config, dipy_sphere)

    def get_segment(self, pos, v_in):
        return None, None, None
    
    def _append_to_line(self, new_pos, new_dir, line, line_dirs, stop_in_tissue=False):
        return None, None, True

def append_to_line(tracker, tissue_key, new_pos, new_dir, line, line_dirs, stop_in_tissue=None):
    if tissue_key in tissue_trackers.keys():
        tracker_f = tracker.tracker_functions[tissue_key]
        if stop_in_tissue is None:
            stop_in_tissue = tracker.tracker_functions[tissue_key].stop_in_tissue
        return tracker_f._append_to_line(new_pos, new_dir, line, line_dirs, stop_in_tissue=stop_in_tissue)
    else:
        return None, None, True

tissue_trackers = {
    "0": background_tissue_tracker,
    "1": wm_tissue_tracker,
    "2": gm_tissue_tracker,
    "3": nuclei_tissue_tracker,
    "4": csf_tissue_tracker
}