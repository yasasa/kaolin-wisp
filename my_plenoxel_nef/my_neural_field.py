# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

# This file is based on https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/wisp/models/nefs/nerf.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math
import copy

from wisp.ops.spc import sample_spc
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.geometric import sample_unif_sphere

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.accelstructs import OctreeAS
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import *

import kaolin.ops.spc as spc_ops

class MyPlenoxelNeuralField(BaseNeuralField):
    """Model for encoding radiance fields (density and plenoptic color)
    """
    def init_embedder(self):
        """Creates positional embedding functions for the position and view direction.
        """
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(frequencies=self.pos_multires,  # 10
                                                                        active=(self.embedder_type == "positional"),
                                                                        input_dim=3)
        self.view_embedder, self.view_embed_dim = get_positional_embedder(frequencies=self.view_multires,  # 4
                                                                         active=(self.embedder_type == "positional"),
                                                                         input_dim=3)
        log.info(f"Position Embed Dim: {self.pos_embed_dim}")
        log.info(f"View Embed Dim: {self.view_embed_dim}")

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders should map from features to output values (such as: rgb, density, sdf, etc), for example:
        """
        # Determine: What is the effective feature dimensions?
        # (are we using concatenation or summation to consolidate features from multiple LODs?)
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        elif self.multiscale_type == 'sum':
            self.effective_feature_dim = self.grid.feature_dim
        else:
            raise NotImplementedError('This neural field supports only concatenation or summation '
                                      'of features from multiple LODs')

        self.input_dim = self.effective_feature_dim + self.view_embed_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        self.decoder = BasicDecoder(input_dim=self.input_dim, output_dim=4,  # RGBA
                                    activation=get_activation_class(self.activation_type), 
                                    bias=True, layer=get_layer_class(self.layer_type), 
                                    num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, skip=[])

    def init_grid(self):
        """ Creates the feature structure this neural field uses, i.e: Octree, Triplane, Hashed grid and so forth.
        The feature grid is queried with coordinate samples during ray tracing / marching.
        The feature grid may also include an occupancy acceleration structure internally to speed up
        tracers.
        """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            
            if self.grid_type == "HashGrid":
                # TODO(ttakikawa): Expose these parameters. 
                # This is still an experimental feature for the most part. It does work however.
                density_decay = 0.6
                min_density = ((0.01 * 512)/np.sqrt(3))

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                #idx = torch.randperm(points.shape[0]) # [:N] to subsample
                res = 2.0**self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples[:,None], ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density
                
                #print(density.mean())
                #print(density.max())
                #print(mask.sum())
                #print(self.grid.occupancy.max())

                _points = points[mask]
                octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
                self.grid.blas.init(octree)
            else:
                raise NotImplementedError

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'my_plenoxel_nerf'

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def calc_sh_basis(self, ray_d, basis_dim=9):
        """Compute spherical harmonic basis for given viewing direction.
        Based on calc_sh from https://github.com/sxyu/svox2/blob/master/svox2/csrc/include/render_util.cuh

        Args:
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            basis_dim (float): basis dimension for spherical harmonics
        
        Returns:
            sh_basis tensor of shape [batch, basis_dim] 
        """

        # SH Coefficients from https://github.com/google/spherical-harmonics
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [1.0925484305920792,
              -1.0925484305920792,
              0.31539156525252005,
              -1.0925484305920792,
              0.5462742152960396]

        sh_basis = torch.zeros((ray_d.shape[0],basis_dim), device=ray_d.device)

        sh_basis[:,0].fill_(C0)
        x, y, z = ray_d[:,0], ray_d[:,1], ray_d[:,2]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        if basis_dim == 9:
            sh_basis[:,4] = C2[0] * xy
            sh_basis[:,5] = C2[1] * yz
            sh_basis[:,6] = C2[2] * (2.0 * zz - xx - yy)
            sh_basis[:,7] = C2[3] * xz
            sh_basis[:,8] = C2[4] * (xx - yy)

        if basis_dim == 4 or basis_dim == 9:
            sh_basis[:,1] = -C1 * y
            sh_basis[:,2] = C1 * z
            sh_basis[:,3] = -C1 * x
        else:
            raise ValueError

        return sh_basis

    def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3] 
                - Density tensor of shape [batch, num_samples, 1]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_rgba_preprocess")
        
        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("rf_rgba_interpolate")

        plenoxel_model = False
        if plenoxel_model:
            # Plenoxel features are a scalar density and a vector of spherical 
            # harmonic coefficients for each color channel. SH of degree 2 is used,
            # requiring 9 coefficients per color channel for 27 coeffs total.
            # ReLU is used to ensure predicted sample colors and densities are 
            # between 0 and 1.
            BASIS_DIM = 9
            if BASIS_DIM == 9:
                assert self.effective_feature_dim == 28, "effective_feature_dim: {}".format(self.effective_feature_dim)
            elif BASIS_DIM == 4:
                assert self.effective_feature_dim == 13, "effective_feature_dim: {}".format(self.effective_feature_dim)
            
            # Density stored as scalar in feature vec
            density = torch.relu(feats[:,self.effective_feature_dim-1]).reshape(batch, num_samples, -1)

            # Extract SH coeffs from feature vec, compute SH basis using viewing 
            # direction, and compute color as sum of basis weighted by coeffs.
            sh_coeffs = feats[:,0:self.effective_feature_dim-1].reshape(-1, 3, BASIS_DIM)
            sh_basis = self.calc_sh_basis(ray_d, basis_dim=BASIS_DIM).reshape(-1, 1, BASIS_DIM)
            timer.check("rf_rgba_sh_basis")
            # TODO: Plenoxels uses ReLU instead of sigmoid, but not sure how they keep color less than 1
            colors = torch.sigmoid(torch.sum(sh_coeffs * sh_basis, dim=2)).reshape(batch, num_samples, -1)
            timer.check("rf_rgba_sh_sumprod")

        else:
            # Optionally concat the positions to the embedding, and also concatenate embedded view directions.
            if self.position_input:
                fdir = torch.cat([feats,
                    self.pos_embedder(coords.reshape(-1, 3)),
                    self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)
            else: 
                fdir = torch.cat([feats,
                    self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)
            timer.check("rf_rgba_embed_cat")
            
            # Decode high-dimensional vectors to RGBA.
            rgba = self.decoder(fdir)
            timer.check("rf_rgba_decode")

            # Colors are values [0, 1] floats
            colors = torch.sigmoid(rgba[...,:3]).reshape(batch, num_samples, -1)

            # Density is [particles / meter], so need to be multiplied by distance
            density = torch.relu(rgba[...,3:4]).reshape(batch, num_samples, -1)
            timer.check("rf_rgba_activation")
        
        return dict(rgb=colors, density=density)

