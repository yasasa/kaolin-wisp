# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import logging
import os
import yaml
import inspect
import torch
import numpy as np
from typing import Dict, List, Any

from wisp.models import nefs
from wisp.models import grids
from wisp import tracers
from wisp import datasets

from wisp.models.grids import BLASGrid, OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.models.nefs import BaseNeuralField, NeuralRadianceField
from wisp.tracers import BaseTracer, PackedRFTracer
from wisp.models.pipeline import Pipeline

# This file contains all the configuration and command-line parsing general to all app

__all__ = [
    'list_modules',
    'register_module',
    'get_module',
    'get_args_for_function',
    'get_grouped_args',
    'parse_args'
]

# str2mod ("str to module") are registered wisp blocks the config parser is aware of, and is able to dynamically load.
# You may register additional options by adding them to the dictionary here.
# ConfigParser expects the following default module categories:
str2mod = {
    'optim': {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()},
    'dataset': {},
    'nef': {},
    'grid': {},
    'tracer': {}
}

def parse_options():
    parser = argparse.ArgumentParser(description='A script for training simple NeRF variants.')
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')

    log_group = parser.add_argument_group('logging')
    log_group.add_argument('--exp-name', type=str,
                           help='Experiment name, unique id for trainers, logs.')
    log_group.add_argument('--log-level', action='store', type=int, default=logging.INFO,
                           help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
    log_group.add_argument('--perf', action='store_true', default=False,
                           help='Use high-level profiling for the trainer.')

    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--dataset-path', type=str,
                            help='Path to the dataset')
    data_group.add_argument('--dataset-num-workers', type=int, default=-1,
                            help='Number of workers for dataset preprocessing, if it supports multiprocessing. '
                                 '-1 indicates no multiprocessing.')
    data_group.add_argument('--dataloader-num-workers', type=int, default=0,
                            help='Number of workers for dataloader.')
    data_group.add_argument('--bg-color', default='white', choices=['white', 'black', 'predict'],
                            help='Background color')
    data_group.add_argument('--multiview-dataset-format', default='standard', choices=['standard', 'rtmv'],
                            help='Data format for the transforms')
    data_group.add_argument('--num-rays-sampled-per-img', type=int, default='4096',
                            help='Number of rays to sample per image')
    data_group.add_argument('--mip', type=int, default=None,
                            help='MIP level of ground truth image')

    grid_group = parser.add_argument_group('grid')
    grid_group.add_argument('--grid-type', type=str, default='OctreeGrid',
                            choices=list_modules('grid'),
                            help='Type of to use, i.e.:'
                                 '"OctreeGrid", "CodebookOctreeGrid", "TriplanarGrid", "HashGrid".'
                                 'Grids are located in `wisp.models.grids`')
    grid_group.add_argument('--interpolation-type', type=str, default='linear', choices=['linear', 'closest'],
                            help='Interpolation type to use for samples within grids.'
                                 'For a 3D grid structure, linear uses trilinear interpolation of 8 cell nodes,'
                                 'closest uses the nearest neighbor.')
    grid_group.add_argument('--blas-type', type=str, default='octree',  # TODO(operel)
                            choices=['octree',],
                            help='Type of acceleration structure to use for fast raymarch occupancy queries.')
    grid_group.add_argument('--multiscale-type', type=str, default='sum', choices=['sum', 'cat'],
                            help='Aggregation of choice for multi-level grids, for features from different LODs.')
    grid_group.add_argument('--feature-dim', type=int, default=32,
                            help='Dimensionality for features stored within the grid nodes.')
    grid_group.add_argument('--feature-std', type=float, default=0.0,
                            help='Grid initialization: standard deviation used for randomly sampling initial features.')
    grid_group.add_argument('--feature-bias', type=float, default=0.0,
                            help='Grid initialization: bias used for randomly sampling initial features.')
    grid_group.add_argument('--base-lod', type=int, default=2,
                            help='Number of levels in grid, which book-keep occupancy but not features.'
                                 'The total number of levels in a grid is `base_lod + num_lod - 1`')
    grid_group.add_argument('--num-lods', type=int, default=1,
                            help='Number of levels in grid, which store concrete features.')
    grid_group.add_argument('--codebook-bitwidth', type=int, default=8,
                            help='For Codebook and HashGrids only: determines the table size as 2**(bitwidth).')
    grid_group.add_argument('--tree-type', type=str, default='geometric', choices=['geometric', 'quad'],
                            help='For HashGrids only: how the resolution of the grid is determined. '
                                 '"geometric" uses the geometric sequence initialization from InstantNGP,'
                                 'where "quad" uses an octree sampling pattern.')
    grid_group.add_argument('--min-grid-res', type=int, default=16,
                            help='For HashGrids only: min grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--max-grid-res', type=int, default=2048,
                            help='For HashGrids only: max grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--prune-min-density', type=float, default=(0.01 * 512) / np.sqrt(3),
                            help='For HashGrids only: Minimum density value for pruning')
    grid_group.add_argument('--prune-density-decay', type=float, default=0.6,
                            help='For HashGrids only: The decay applied on the density every pruning')
    grid_group.add_argument('--blas-level', type=float, default=7,
                            help='For HashGrids only: Determines the number of levels in the acceleration structure '
                                 'used to track the occupancy status (bottom level acceleration structure).')

    nef_group = parser.add_argument_group('nef')
    nef_group.add_argument('--pos-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode input coordinates'
                                'or view directions.')
    nef_group.add_argument('--view-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode view direction')
    nef_group.add_argument('--position-input', type=bool, default=False,
                           help='If True, position coords will be concatenated to the '
                                'features / positional embeddings when fed into the decoder.')
    nef_group.add_argument('--pos-multires', type=int, default=10,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of input coordinates')
    nef_group.add_argument('--view-multires', type=int, default=4,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of view direction')
    nef_group.add_argument('--layer-type', type=str, default='none',
                           choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    nef_group.add_argument('--activation-type', type=str, default='relu',
                           choices=['relu', 'sin'])
    nef_group.add_argument('--hidden-dim', type=int, help='MLP Decoder of neural field: width of all hidden layers.')
    nef_group.add_argument('--num-layers', type=int, help='MLP Decoder of neural field: number of hidden layers.')

    tracer_group = parser.add_argument_group('tracer')
    tracer_group.add_argument('--raymarch-type', type=str, choices=['ray', 'voxel'], default='ray',
                              help='Marching algorithm to use when generating samples along rays in tracers.'
                                   '`ray` samples fixed amount of randomized `num_steps` along the ray.'
                                   '`voxel` samples `num_steps` samples in each cell the ray intersects.')
    tracer_group.add_argument('--num-steps', type=int, default=1024,
                              help='Number of samples to generate along traced rays. See --raymarch-type for '
                                   'algorithm used to generate the samples.')

    trainer_group = parser.add_argument_group('trainer')
    trainer_group.add_argument('--epochs', type=int, default=250,
                               help='Number of epochs to run the training.')
    trainer_group.add_argument('--batch-size', type=int, default=512,
                               help='Batch size for the training.')
    trainer_group.add_argument('--resample', action='store_true',
                               help='Resample the dataset after every epoch.')
    trainer_group.add_argument('--only-last', action='store_true',
                               help='Train only last LOD.')
    trainer_group.add_argument('--resample-every', type=int, default=1,
                               help='Resample every N epochs')
    trainer_group.add_argument('--model-format', type=str, default='full', choices=['full', 'state_dict'],
                               help='Format in which to save models.')
    trainer_group.add_argument('--pretrained', type=str,
                               help='Path to pretrained model weights.')
    trainer_group.add_argument('--save-as-new', action='store_true',
                               help='Save the model at every epoch (no overwrite).')
    trainer_group.add_argument('--save-every', type=int, default=5,
                               help='Save the model at every N epoch.')
    trainer_group.add_argument('--render-tb-every', type=int, default=5,
                               help='Render every N epochs')
    trainer_group.add_argument('--log-tb-every', type=int, default=5, # TODO (operel): move to logging
                               help='Render to tensorboard every N epochs')
    trainer_group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                               help='Log file directory for checkpoints.')
    trainer_group.add_argument('--prune-every', type=int, default=-1,
                               help='Prune every N epochs')
    trainer_group.add_argument('--grow-every', type=int, default=-1,
                               help='Grow network every X epochs')
    trainer_group.add_argument('--growth-strategy', type=str, default='increase',
                               choices=['onebyone',      # One by one trains one level at a time.
                                        'increase',      # Increase starts from [0] and ends up at [0,...,N]
                                        'shrink',        # Shrink strats from [0,...,N] and ends up at [N]
                                        'finetocoarse',  # Fine to coarse starts from [N] and ends up at [0,...,N]
                                        'onlylast'],     # Only last starts and ends at [N]
                               help='Strategy for coarse-to-fine training')
    trainer_group.add_argument('--valid-only', action='store_true',
                               help='Run validation only (and do not run training).')
    trainer_group.add_argument('--valid-every', type=int, default=-1,
                               help='Frequency of running validation.')
    trainer_group.add_argument('--random-lod', action='store_true',
                               help='Use random lods to train.')
    trainer_group.add_argument('--wandb-project', type=str, default=None,
                               help='Weights & Biases Project')
    trainer_group.add_argument('--wandb-run-name', type=str, default=None,
                               help='Weights & Biases Run Name')
    trainer_group.add_argument('--wandb-entity', type=str, default=None,
                               help='Weights & Biases Entity')
    trainer_group.add_argument('--wandb-viz-nerf-angles', type=int, default=20,
                               help='Number of Angles to visualize a scene on Weights & Biases. '
                                    'Set this to 0 to disable 360 degree visualizations.')
    trainer_group.add_argument('--wandb-viz-nerf-distance', type=int, default=3,
                               help='Distance to visualize Scene from on Weights & Biases')

    optimizer_group = parser.add_argument_group('optimizer')
    optimizer_group.add_argument('--optimizer-type', type=str, default='adam',
                                 choices=list_modules('optim'),
                                 help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                                      'and fused optimizers from `apex`, if apex is installed.')
    optimizer_group.add_argument('--lr', type=float, default=0.001,
                                 help='Base optimizer learning rate.')
    optimizer_group.add_argument('--eps', type=float, default=1e-8,
                                 help='Eps value for numerical stability.')
    optimizer_group.add_argument('--weight-decay', type=float, default=0,
                                 help='Weight decay, applied only to decoder weights.')
    optimizer_group.add_argument('--grid-lr-weight', type=float, default=100.0,
                                 help='Relative learning rate weighting applied only for the grid parameters'
                                      '(e.g. parameters which contain "grid" in their name)')
    optimizer_group.add_argument('--rgb-loss', type=float, default=1.0,
                                 help='Weight of rgb loss')

    # Evaluation renderer (definitions do not affect interactive renderer)
    offline_renderer_group = parser.add_argument_group('renderer')
    offline_renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512],
                                        help='Width/height to render at.')
    offline_renderer_group.add_argument('--render-batch', type=int, default=0,
                                        help='Batch size (in number of rays) for batched rendering.')
    offline_renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8],
                                        help='Camera origin.')
    offline_renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0],
                                        help='Camera look-at/target point.')
    offline_renderer_group.add_argument('--camera-fov', type=float, default=30,
                                        help='Camera field of view (FOV).')
    offline_renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp',
                                        help='Camera projection.')
    offline_renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10],
                                        help='Camera clipping bounds.')

    return parser


def list_modules(module_type) -> List[str]:
    """ Returns a list of all available modules from a certain category.
    Args:
        type: a str from the following categories: ['nef', 'grid', 'tracer', 'dataset', 'optim']
    """
    return list(str2mod[module_type].keys())


def register_module(module_type, name, mod):
    """Register module to be used with config parser.
    Users should use this class to load their classes by name.
    """
    if module_type not in str2mod:
        raise ValueError(f"'{module_type}' is an unknown type")

    if name in str2mod[module_type]:
        raise KeyError(f"'{name}' already exist in type '{module_type}'")
    str2mod[module_type][name] = mod


def get_module(name, module_type=None):
    """Get module class by name, assuming it was registered with `register_module`.'"""
    types_to_check = []
    if module_type is None:
        types_to_check = str2mod
    else:
        if module_type not in str2mod:
            raise ValueError(f"'{module_type}' is an unknown type")
        types_to_check.append(module_type)

    for t in types_to_check:
        if name in str2mod[t]:
            return str2mod[t][name]

    raise ValueError(f"'{name}' is not a known module for any of the types '{types_to_check}'. "
                     f"registered modules are '{str2mod[module_type].keys()}'")


def get_args_for_function(args, func):
    """ Given a func (for example an __init__(..) function or from_X(..)), and also the parsed args,
    return the subset of args that func expects and args contains. """
    has_kwargs = inspect.getfullargspec(func).varkw != None
    if has_kwargs:
        collected_args = vars(args)
    else:
        parameters = dict(inspect.signature(func).parameters)
        collected_args = {a: getattr(args, a) for a in parameters if hasattr(args, a)}
    return collected_args


def get_grouped_args(parser, args) -> Dict[str, Any]:
    """Group args to a grouped hierarchy.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.

    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))
    return args_dict


# -- Register all wisp library modules here -- this makes them loadable by specifying the class name to get_module() --
for name in dir(datasets):
    mod = getattr(datasets, name)
    if isinstance(mod, type) and \
            issubclass(mod, torch.utils.data.Dataset) and \
            mod != torch.utils.data.Dataset:
        register_module('dataset', name, mod)

for name in dir(nefs):
    mod = getattr(nefs, name)
    if isinstance(mod, type) and \
            issubclass(mod, nefs.BaseNeuralField) and \
            mod != nefs.BaseNeuralField:
        register_module('nef', name, mod)

for name in dir(grids):
    mod = getattr(grids, name)
    if isinstance(mod, type) and \
            issubclass(mod, grids.BLASGrid) and \
            mod != grids.BLASGrid:
        register_module('grid', name, mod)

for name in dir(tracers):
    mod = getattr(tracers, name)
    if isinstance(mod, type) and \
            issubclass(mod, tracers.BaseTracer) and \
            mod != tracers.BaseTracer:
        register_module('tracer', name, mod)

try:
    import apex

    for m in dir(apex.optimizers):
        if m[0].isupper():
            register_module('optim', m.lower(), getattr(apex.optimizers, m))
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Cannot import apex for fused optimizers")


def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)

    defaults_dict = {}
    defaults_dict_args = {}

    # Load the parent config if it exists
    parent_config_path = config_dict.pop("parent", None)

    if parent_config_path is not None:
        if not os.path.isabs(parent_config_path):
            parent_config_path = os.path.join(os.path.split(config_path)[0], parent_config_path)
        with open(parent_config_path) as f:
            parent_config_dict = yaml.safe_load(f)
        if "parent" in parent_config_dict.keys():
            raise Exception("Hierarchical configs of more than 1 level deep are not allowed.")
        for key in parent_config_dict:
            for field in parent_config_dict[key]:
                if field not in list_of_valid_fields:
                    raise ValueError(
                        f"ERROR: {field} is not a valid option. Check for typos in the config."
                    )
                if isinstance(parent_config_dict[key][field], dict):
                    defaults_dict_args[field] = parent_config_dict[key][field]
                    defaults_dict[field] = None
                else:
                    defaults_dict[field] = parent_config_dict[key][field]

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            if isinstance(config_dict[key][field], dict):
                defaults_dict_args[field] = config_dict[key][field]
                defaults_dict[field] = None
            else:
                defaults_dict[field] = config_dict[key][field]

    parser.set_defaults(**defaults_dict)
    return defaults_dict_args


def parse_args(parser, args=None) -> argparse.Namespace:
    """Parses args by priority into a flat configuration.
    The various options take the following precedence:
    1. CLI args, explicitly specified
    2. YAML configuration, defined with `--config <PATH>.yaml`
    3. argparse defaults

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.

    Returns:
        args    : The parsed arguments, as a flat configuration.
    """

    no_args = args is None
    if no_args:
        args = parser.parse_args()
    defaults_dict = dict()

    if args.config is not None:
        config_defaults_dict = parse_yaml_config(args.config, parser)
        for key, val in config_defaults_dict.items():
            if key in defaults_dict:
                defaults_dict[key].update(val)
            else:
                defaults_dict[key] = val
        if no_args:
            args = parser.parse_args()
        else:
            args = parser.parse_args("")

    for key, val in defaults_dict.items():
        cmd_line_val = getattr(args, key)
        if cmd_line_val is not None:
            val.update(cmd_line_val)
        setattr(args, key, val)

    return args

def load_grid(args, dataset: torch.utils.data.Dataset) -> BLASGrid:
    """ Wisp's implementation of NeRF uses feature grids to improve the performance and quality (allowing therefore,
    interactivity).
    This function loads the feature grid to use within the neural pipeline.
    Grid choices are interesting to explore, so we leave the exact backbone type configurable,
    and show how grid instances may be explicitly constructed.
    Grids choices, for example, are: OctreeGrid, TriplanarGrid, HashGrid, CodebookOctreeGrid
    See corresponding grid constructors for each of their arg details.
    """
    grid = None
    # Optimization: For octrees based grids, if dataset contains depth info, initialize only cells known to be occupied
    if dataset is not None:
        has_depth_supervision = getattr(dataset, "coords", None) is not None
    else:
        assert args.grid_type not in ["OctreeGrid", "CodebookOctreeGrid"], "dataset must be provided"

    if args.grid_type == "OctreeGrid":
        if has_depth_supervision:
            grid = OctreeGrid.from_pointcloud(
                pointcloud=dataset.coords,
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
            )
        else:
            grid = OctreeGrid.make_dense(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
            )
    elif args.grid_type == "CodebookOctreeGrid":
        if has_depth_supervision:
            grid = CodebookOctreeGrid.from_pointcloud(
                pointcloud=dataset.coords,
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth
            )
        else:
            grid = CodebookOctreeGrid.make_dense(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth
            )
    elif args.grid_type == "TriplanarGrid":
        grid = TriplanarGrid(
            feature_dim=args.feature_dim,
            base_lod=args.base_lod,
            num_lods=args.num_lods,
            interpolation_type=args.interpolation_type,
            multiscale_type=args.multiscale_type,
            feature_std=args.feature_std,
            feature_bias=args.feature_bias,
        )
    elif args.grid_type == "HashGrid":
        # "geometric" - determines the resolution of the grid using geometric sequence initialization from InstantNGP,
        if args.tree_type == "geometric":
            grid = HashGrid.from_geometric(
                feature_dim=args.feature_dim,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                min_grid_res=args.min_grid_res,
                max_grid_res=args.max_grid_res,
                blas_level=args.blas_level
            )
        # "quad" - determines the resolution of the grid using an octree sampling pattern.
        elif args.tree_type == "octree":
            grid = HashGrid.from_octree(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                blas_level=args.blas_level
            )
    else:
        raise ValueError(f"Unknown grid_type argument: {args.grid_type}")
    return grid


def load_neural_field(args, dataset: torch.utils.data.Dataset) -> BaseNeuralField:
    """ Creates a "Neural Field" instance which converts input coordinates to some output signal.
    Here a NeuralRadianceField is created, which maps 3D coordinates (+ 2D view direction) -> RGB + density.
    The NeuralRadianceField uses spatial feature grids internally for faster feature interpolation and raymarching.
    """
    grid = load_grid(args=args, dataset=dataset)
    nef = NeuralRadianceField(
        grid=grid,
        pos_embedder=args.pos_embedder,
        view_embedder=args.view_embedder,
        position_input=args.position_input,
        pos_multires=args.pos_multires,
        view_multires=args.view_multires,
        activation_type=args.activation_type,
        layer_type=args.layer_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        prune_density_decay=args.prune_density_decay,   # Used only for grid types which support pruning
        prune_min_density=args.prune_min_density,        # Used only for grid types which support pruning
        bg_color=args.bg_color
    )
    return nef


def load_tracer(args) -> BaseTracer:
    """ Wisp "Tracers" are responsible for taking input rays, marching them through the neural field to render
    an output RenderBuffer.
    Wisp's implementation of NeRF uses the PackedRFTracer to trace the neural field:
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is employed within the training loop, and is responsible for making use of the neural field's
    grid to generate samples and decode them to pixel values.
    """
    tracer = PackedRFTracer(
        raymarch_type=args.raymarch_type,   # Chooses the ray-marching algorithm
        num_steps=args.num_steps,           # Number of steps depends on raymarch_type
        bg_color=args.bg_color
    )
    return tracer


def load_neural_pipeline(args, dataset, device) -> Pipeline:
    """ In Wisp, a Pipeline comprises of a neural field + a tracer (the latter is optional in some cases).
    Together, they form the complete pipeline required to render a neural primitive from input rays / coordinates.
    """
    nef = load_neural_field(args=args, dataset=dataset)
    tracer = load_tracer(args=args)
    pipeline = Pipeline(nef=nef, tracer=tracer)
    if args.pretrained:
        print("Loading pretrained model from {}". format(args.pretrained))
        if args.model_format == "full":
            pipeline = torch.load(args.pretrained)
        else:
            pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)
    return pipeline