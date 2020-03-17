# -*- coding: utf-8 -*-

from scilpy.tracking.tissue_tracker import tissue_trackers
import numpy as np
from scilpy.tracking.tools import sample_distribution

class abstractTracker(object):

    def __init__(self, atlas, odf_dataset, basis, config, param):
        self.atlas = atlas
        self.odf = odf_dataset
        self.basis = basis
        self.config = config
        self.max_nbr_points = param['max_nbr_pts']
        self.step_size = param['step_size']
        self.sf_threshold = param['sf_threshold']
        self.sf_threshold_init = param['sf_threshold_init']
        unique_atlas_val = np.unique(atlas.data)
        self.tracker_functions = {}
        for label in unique_atlas_val:
            tracker_f = tissue_trackers[str(int(label))]
            self.tracker_functions[str(int(label))] = tracker_f(self, self.odf, self.basis, self.sf_threshold, self.sf_threshold_init, self.config, dipy_sphere='symmetric724')


    def initialize(self, pos):
        """
        Initialise the tracking at position pos. Initial tracking directions are
        picked, the propagete_foward() and propagate_backward() functions
        could then be call.
        return True if initial tracking directions are found.
        """
        self.init_pos = pos
        self.forward_pos = pos
        self.backward_pos = pos
        tissue_tracker = self.get_tissue_tracker_function(pos)
        self.forward_dir, self.backward_dir = tissue_tracker.tracking_field.get_init_direction(
            pos)
        return self.forward_dir is not None and self.backward_dir is not None
    
    def get_tissue_tracker_function(self, pos):
        tissue = int(self.atlas.getPositionValue(pos[0], pos[1], pos[2]))
        return self.tracker_functions[str(tissue)]

    def propagate(self, pos, v_in):
        """
        return tuple. The new tracking direction and the updated position.
        If no valid tracking direction are available, v_in is choosen.
        """
        tissue_tracker = self.get_tissue_tracker_function(pos)
        newpos, newdir, is_valid = tissue_tracker.get_segment(pos, v_in)
        if is_valid:
            return newpos, newdir, str(int(self.atlas.getPositionValue(newpos[0], newpos[1], newpos[2])))
        else:
            return None, None, None


    def isPositionInBound(self, pos):
        """
        Test if the streamline point is inside the boundary of the image.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        tissue_tracker = self.get_tissue_tracker_function(pos)
        return tissue_tracker.tracking_field.dataset.isPositionInBound(*pos)

    def get_direction(self, pos, v_in):
        """
        return the next tracking direction, given the current position pos
        and the previous direction v_in. This direction must respect tracking
        constraint defined in the trackingField.
        """
        scilpy.utils.abstract()


class probabilisticTracker(abstractTracker):

    def __init__(self, atlas, odf_dataset, basis, config, param):
        super(probabilisticTracker, self).__init__(
             atlas, odf_dataset, basis, config, param)

    def get_direction(self, pos, v_in):
        """
        return a direction drawn from the distribution weighted with
        the spherical function.
        None if the no valid direction are available.
        """
        tissue_tracker = self.get_tissue_tracker_function(pos)
        sf, directions = tissue_tracker.tracking_field.get_tracking_SF(pos, v_in)
        if np.sum(sf) > 0:
            return directions[sample_distribution(sf)]
        return None