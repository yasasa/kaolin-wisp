# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np


class SampleRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples, weights = None):
        self.num_samples = num_samples

    def __call__(self, inputs):
        if 'depths' in inputs:
            depths  = inputs['depths'].clone().squeeze()
            depths[depths < 1e-4] = 30.
            weights = 1. / depths
            weights = weights / weights.sum()
          # ray_idx = np.random.choice(weights.shape[0], self.num_samples, replace=False, p=weights.numpy())
          # ray_idx = torch.from_numpy(ray_idx).to(inputs['imgs'].device).long()
            ray_idx = torch.multinomial(weights, self.num_samples, replacement=True).to(inputs['imgs'].device)
        else:
            ray_idx = torch.randint(0, inputs['imgs'].shape[0], [self.num_samples],
            device=inputs['imgs'].device)

        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()
        out['imgs'] = inputs['imgs'][ray_idx].contiguous()
        if 'depths' in inputs:
            out['depths'] = inputs['depths'][ray_idx].contiguous()
        return out
