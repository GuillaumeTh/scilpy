# -*- coding: utf-8 -*-

import random
import os

import numpy as np
from scipy.spatial import distance

class TissueConfigurator(object):
    # Renomer stop_in_nuclei -> seed properties.
    def __init__(self, config, id, stop_in_nuclei=False):
        self.id = id
        self.theta = config[id]["theta"]
        self.step_size = config[id]["step_size"]
        self.algo = config[id]["algo"]
        self.rk_order = config[id]["rk_order"]
        self.stop_in_nuclei = stop_in_nuclei
        if "max_nb_ending_steps" in config[id]:
            self.max_nb_ending_steps = config[id]["max_nb_ending_steps"]
        else:
            self.max_nb_ending_steps = 0

        if "max_consecutive_steps" in config[id]:
            self.max_consecutive_steps = config[id]["max_consecutive_steps"]
        else:
            self.max_consecutive_steps = 1

    def updatePropagator(self, propagator):
        propagator.update_propagator(step_size = self.step_size,
                                     theta = self.theta,
                                     algo = self.algo,
                                     rk_order = self.rk_order)
        return propagator

    def can_continue(self, line, mask):
        return True

    def is_valid_endpoint(self):
        return False

    def finalize_streamline(self, line, last_dir, tracker):
        return []

class BackgroundTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def can_continue(self, line, mask):
        return False

class WmTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

class GmTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def can_continue(self, line, mask):
        history = []
        for i in line[-self.max_consecutive_steps:]:
            history.append(mask.voxmm_to_value(*i,
                                    origin='corner'))
        if len(np.unique(history)) == 1:
            return False

        value = mask.voxmm_to_value(*line[-1],
                                    origin='corner', interpolator="trilinear")
        difference = value / self.id
        rand_value = random.random()
        if rand_value < difference:
            return True
        else:
            return False

    def is_valid_endpoint(self):
        return True

    def finalize_streamline(self, line, last_dir, tracker):
        if self.is_valid_endpoint():
            random.seed(np.uint32(hash((tuple(line[-1]), tracker.rng_seed))))
            nb_ending_steps = self.max_nb_ending_steps
            if self.max_nb_ending_steps:
                nb_ending_steps = random.randint(1, self.max_nb_ending_steps)
            for i in range(nb_ending_steps):
                # Make a last step in the last direction
                # Ex: if mask is WM, reaching GM a little more.
                new_pos, new_dir, _ = tracker.propagator.propagate(
                    line[-1], last_dir)
                line.append(new_pos)
                last_dir = new_dir

                new_id = int(tracker.mask.voxmm_to_value(*line[-1],
                                origin=tracker.origin))
                if new_id != id:
                    line.pop()
                    break
            return line
        else:
            return []

class NucleiTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id, stop_in_nuclei)

    def can_continue(self, line, mask):
        if self.stop_in_nuclei:
            return False
        
        history = []
        for i in line[-self.max_consecutive_steps:]:
            history.append(mask.voxmm_to_value(*i,
                                    origin='corner'))
        if len(np.unique(history)) == 1:
            return False

        value = mask.voxmm_to_value(*line[-1],
                                    origin='corner', interpolator="trilinear")
        difference = np.abs(value - self.id)
        rand_value = random.random()
        if rand_value < difference:
            return True
        else:
            return False


    def is_valid_endpoint(self):
        if self.stop_in_nuclei:
            return True
        else:
            return False

    def finalize_streamline(self, line, last_dir, tracker):
        if self.is_valid_endpoint():
            random.seed(np.uint32(hash((tuple(line[-1]), tracker.rng_seed))))
            nb_ending_steps = self.max_nb_ending_steps
            if self.max_nb_ending_steps:
                nb_ending_steps = random.randint(1, self.max_nb_ending_steps)
            for i in range(nb_ending_steps):
                # Make a last step in the last direction
                # Ex: if mask is WM, reaching GM a little more.
                new_pos, new_dir, _ = tracker.propagator.propagate(
                    line[-1], last_dir)
                line.append(new_pos)
                last_dir = new_dir

                new_id = int(tracker.mask.voxmm_to_value(*line[-1],
                                origin=tracker.origin))
                if new_id != id:
                    line.pop()
                    return line
            return line
        else:
            return []

class CsfTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def can_continue(self, line, mask):
        history = []
        for i in line[-self.max_consecutive_steps:]:
            history.append(mask.voxmm_to_value(*i,
                                    origin='corner'))
        if len(np.unique(history)) == 1:
            return False

        value = mask.voxmm_to_value(*line[-1],
                                    origin='corner', interpolator="trilinear")
        difference = value / self.id
        rand_value = random.random()
        if rand_value < difference:
            return True
        else:
            return False

class GmSurfaceTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def can_continue(self, line, mask):
        history = []
        for i in line[-self.max_consecutive_steps:]:
            history.append(mask.voxmm_to_value(*i,
                                    origin='corner'))
        if len(np.unique(history)) == 1:
            return False

        value = mask.voxmm_to_value(*line[-1],
                                    origin='corner', interpolator="trilinear")
        difference = value / self.id
        rand_value = random.random()
        if rand_value < difference:
            return True
        else:
            return False

    def is_valid_endpoint(self):
        return True

    def finalize_streamline(self, line, last_dir, tracker):
        if self.is_valid_endpoint():
            random.seed(np.uint32(hash((tuple(line[-1]), tracker.rng_seed))))
            if self.max_nb_ending_steps:
                nb_ending_steps = random.randint(1, self.max_nb_ending_steps)
            else:
                nb_ending_steps = 1
            vox_idx = tuple(tracker.mask.voxmm_to_idx(*line[-1], tracker.origin).astype(np.int))
            normal_idx = get_closest_vertex(line[-1], vox_idx, tracker.vertices)
            if normal_idx is not None:
                last_dir = -1 * tracker.normals[vox_idx][normal_idx]
                count = 0
                while len(line) > 1 and count < 3:
                    line.pop()
                    count += 1
                for _ in range(count):
                    # Make a last step in the last direction
                    # Ex: if mask is WM, reaching GM a little more.
                    new_pos = line[-1] + self.step_size * np.array(last_dir)
                    # new_pos, new_dir, _ = tracker.propagator.propagate(
                    #     line[-1], np.array(last_dir))
                    line.append(new_pos)
                    # last_dir = last_dir

                    new_id = int(tracker.mask.voxmm_to_value(*line[-1],
                                    origin=tracker.origin))
                    if new_id != self.id:
                        line.pop()
                        return line
                return line
            else:
                return []

def get_tissue_configurator(config, id, stop_in_nuclei) -> TissueConfigurator:
    return tissue_dict[id](config, id, stop_in_nuclei)


def support_surface(id) -> bool:
    if id in [6]:
        return True
    return False

def get_closest_vertex(pos, idx, vertices):
    if idx in vertices:
        #On prend le array de candidats
        canditates = vertices[idx]
        if len(canditates) > 1:
            min_dist = 99
            for id, i in enumerate(canditates):
                cur_dist = distance.euclidean(pos, i)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_idx = id
            return min_idx
        else:
            return 0
    return None

tissue_dict = {0: BackgroundTissue,
               1: WmTissue,
               2: GmTissue,
               3: NucleiTissue,
               4: CsfTissue,
               6: GmSurfaceTissue}