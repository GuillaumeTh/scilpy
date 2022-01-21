# -*- coding: utf-8 -*-

import random

import numpy as np

class TissueConfigurator(object):
    # Renomer stop_in_nuclei -> seed properties.
    def __init__(self, config, id, stop_in_nuclei=False):
        self.id = id
        self.theta = config[id]["theta"]
        self.step_size = config[id]["step_size"]
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
        return propagator

    def can_continue(self, line, mask):
        return True

    def is_valid_endpoint(self):
        raise NotImplementedError

class BackgroundTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def can_continue(self, line, mask):
        return False

    def is_valid_endpoint(self):
        return False

class WmTissue(TissueConfigurator):

    def __init__(self, config, id, stop_in_nuclei):
        super().__init__(config, id)

    def is_valid_endpoint(self):
        return False

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
        # print(rand_value, difference, value, self.id, *line[-1])
        if rand_value < difference:
            return True
        else:
            return False

    def is_valid_endpoint(self):
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

def get_tissue_configurator(config, id, stop_in_nuclei) -> TissueConfigurator:
    return tissue_dict[id](config, id, stop_in_nuclei)

tissue_dict = {0: BackgroundTissue,
               1: WmTissue,
               2: GmTissue,
               3: NucleiTissue,
               4: CsfTissue,
               6: GmSurfaceTissue}