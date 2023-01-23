# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from wisp.cuda_guard import setup_cuda_context
setup_cuda_context()  # Must be called before any torch operations take place

import os
import argparse
import logging
import numpy as np
import torch
from wisp.app_utils import default_log_setup, args_to_log_format
import wisp.config_parser as config_parser
from wisp.framework import WispState
from wisp.datasets import MultiviewDataset
from wisp.datasets.transforms import SampleRays
from wisp.trainers import BaseTrainer, MultiviewTrainer


def parse_args():
    """Wisp mains define args per app.
    Args are collected by priority: cli args > config yaml > argparse defaults
    For convenience, args are divided into groups.
    """
    parser = config_parser.parse_options()

    # Parse CLI args & config files
    args = config_parser.parse_args(parser)

    # Override some definitions for interactive app, such as validation logic and default data background color
    if is_interactive():
        args.bg_color = 'black'
        args.save_every = -1
        args.render_tb_every = -1
        args.valid_every = -1

    # Also obtain args as grouped hierarchy, useful for, i.e., logging
    args_dict = config_parser.get_grouped_args(parser, args)
    return args, args_dict


def load_dataset(args) -> torch.utils.data.Dataset:
    """ Loads a multiview dataset comprising of pairs of images and calibrated cameras.
    The types of supported datasets are defined by multiview_dataset_format:
    'standard' - refers to the standard NeRF format popularized by Mildenhall et al. 2020,
                 including additions to the metadata format added by Muller et al. 2022.
    'rtmv' - refers to the dataset published by Tremblay et. al 2022,
            "RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis".
            This dataset includes depth information which allows for performance improving optimizations in some cases.
    """
    transform = SampleRays(num_samples=args.num_rays_sampled_per_img)
    train_dataset = MultiviewDataset(dataset_path=args.dataset_path,
                                     multiview_dataset_format=args.multiview_dataset_format,
                                     mip=args.mip,
                                     bg_color=args.bg_color,
                                     dataset_num_workers=args.dataset_num_workers,
                                     transform=transform)
    return train_dataset


def load_trainer(pipeline, train_dataset, device, scene_state, args, args_dict) -> BaseTrainer:
    """ Loads the NeRF trainer.
    The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
    - Headless, which will run the train() function until all training steps are exhausted.
    - Interactive mode, which uses the gui. In this case, an OptimizationApp uses events to prompt the trainer to
      take training steps, while also taking care to render output to users (see: iterate()).
      In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
    """
    # args.optimizer_type is the name of some optimizer class (from torch.optim or apex),
    # Wisp's config_parser is able to pick this app's args with corresponding names to the optimizer constructor args.
    # The actual construction of the optimizer instance happens within the trainer.
    optimizer_cls = config_parser.get_module(name=args.optimizer_type)
    optimizer_params = config_parser.get_args_for_function(args, optimizer_cls)

    trainer = MultiviewTrainer(pipeline=pipeline,
                               dataset=train_dataset,
                               num_epochs=args.epochs,
                               batch_size=args.batch_size,
                               optim_cls=optimizer_cls,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               grid_lr_weight=args.grid_lr_weight,
                               optim_params=optimizer_params,
                               log_dir=args.log_dir,
                               device=device,
                               exp_name=args.exp_name,
                               info=args_to_log_format(args_dict),
                               extra_args=vars(args),
                               render_tb_every=args.render_tb_every,
                               save_every=args.save_every,
                               scene_state=scene_state,
                               trainer_mode='validate' if args.valid_only else 'train',
                               using_wandb=args.wandb_project is not None)
    return trainer


def load_app(args, scene_state, trainer):
    """ Used only in interactive mode. Creates an interactive app, which employs a renderer which displays
    the latest information from the trainer (see: OptimizationApp).
    The OptimizationApp can be customized or further extend to support even more functionality.
    """
    if not is_interactive():
        logging.info("Running headless. For the app, set $WISP_HEADLESS=0.")
        return None  # Interactive mode is disabled
    else:
        from wisp.renderer.app.optimization_app import OptimizationApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
        app = OptimizationApp(wisp_state=scene_state,
                              trainer_step_func=trainer.iterate,
                              experiment_name="wisp trainer")
        return app


def is_interactive() -> bool:
    """ Returns True if interactive mode with gui is on, False is HEADLESS mode is forced """
    return os.environ.get('WISP_HEADLESS') != '1'


if __name__ == "__main__":
    args, args_dict = parse_args()  # Obtain args by priority: cli args > config yaml > argparse defaults
    default_log_setup(args.log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = load_dataset(args=args)
    pipeline = config_parser.load_neural_pipeline(args=args, dataset=train_dataset, device=device)
    scene_state = WispState()   # Joint trainer / app state
    trainer = load_trainer(pipeline=pipeline, train_dataset=train_dataset, device=device, scene_state=scene_state,
                        args=args, args_dict=args_dict)
    app = load_app(args=args, scene_state=scene_state, trainer=trainer)

    if app is not None:
        app.run()  # Run in interactive mode
    else:
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()  # Run in headless mode
