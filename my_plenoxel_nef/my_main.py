# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


def parse_args():
    from wisp.config_parser import parse_options, argparse_to_str

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app_utils.add_log_level_flag(parser)

    # Add custom args if needed for app
    app_group = parser.add_argument_group('app')

    args, args_str = argparse_to_str(parser)
    return args, args_str


def create_trainer(args, scene_state):
    """ Create the trainer according to config args """
    from wisp.config_parser import get_modules_from_config, get_optimizer_from_config
    pipeline, train_dataset, device = get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                           optim_cls, args.lr, args.weight_decay,
                                           args.grid_lr_weight, optim_params, args.log_dir, device,
                                           exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                           render_every=args.render_every, save_every=args.save_every,
                                           scene_state=scene_state)
    return trainer


def create_app(scene_state, trainer):
    """ Create the interactive app running the renderer & trainer """
    from template_app import TemplateApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
    interactive_app = TemplateApp(wisp_state=scene_state, background_task=trainer.iterate, window_name="Template App")
    return interactive_app


if __name__ == "__main__":
    # Must be called before any torch operations take place
    from app.cuda_guard import setup_cuda_context
    setup_cuda_context()

    import os
    import app.app_utils as app_utils
    import logging as log
    from wisp.framework import WispState

    # Register any newly added user classes before running the config parser
    # Registration ensures the config parser knows about these classes and is able to dynamically create them.
    from wisp.config_parser import register_class
    from my_neural_field import MyPlenoxelNeuralField
    register_class(MyPlenoxelNeuralField, 'MyPlenoxelNeuralField')
    from wisp.trainers import MultiviewTrainer
    #from template_trainer import TemplateTrainer
    #register_class(TemplateTrainer, 'TemplateTrainer')

    # Parse config yaml and cli args
    args, args_str = parse_args()
    app_utils.default_log_setup(args.log_level)

    # Create the state object, shared by all wisp components
    scene_state = WispState()

    # Create the trainer
    trainer = create_trainer(args, scene_state)

    #print(scene_state.graph.cameras.keys())

    TEST_MULTI_OBJ = True
    if TEST_MULTI_OBJ:
        import torch
        from wisp.renderer.core import RendererCore
        import wisp.renderer.core.renderers
        from wisp.renderer.core.api import add_to_scene_graph
        from wisp.ops.image import write_png, write_exr

        height, width = 360, 640
        core = RendererCore(scene_state)
        core.resize_canvas(height=height, width=width)
        core.set_full_resolution()
        core.redraw()          # Call this every time you add / remove an object

        ##### second object

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if args.pretrained:
        #     if args.model_format == "full":
        #         pipeline = torch.load(args.pretrained)
        #     else:
        #         pipeline.load_state_dict(torch.load(args.pretrained))
        obj_pretrain_path = "/home/salar/RVL/repos/kaolin-wisp/_results/logs/runs/test-ngp-nerf/20220822-231128/model.pth"
        pipeline2 = torch.load(obj_pretrain_path)
        pipeline2.to(device)
        add_to_scene_graph(scene_state, "object2", pipeline2)
        core.redraw()          # Call this every time you add / remove an object

        #scene_state.graph.visible_objects['exp_tmp'] = False
        print(scene_state.graph.visible_objects)

        ##### change camera

        #camera = core.camera
        camera = scene_state.graph.cameras['0_00000']
        scene_state.renderer.selected_camera = camera
        print(camera)


        #####

        rb = core.render()     # Obtain a render buffer
        # See evaluate_metrics() in multiview_trainer.py for examples on how write_png and write_exr are used
        
        #rb = rb.reshape(*img.shape[:2], -1)
                
        rb.view = None
        rb.hit = None

        exrdict = rb.cpu().exr_dict()
        img_out = rb.cpu().image().byte().rgb.numpy()
        print(img_out.shape)

        # out_name = f"{idx}"
        # if name is not None:
        #     out_name += "-" + name

        #write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
        # write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb.numpy())
        write_exr("tmp.exr", exrdict)
        write_png("tmp.png", img_out)

        exit()
            

    if not os.environ.get('WISP_HEADLESS') == '1':
        interactive_app = create_app(scene_state, trainer)
        interactive_app.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()
