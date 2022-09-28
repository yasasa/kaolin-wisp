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

    TEST_MULTI_OBJ = True
    if TEST_MULTI_OBJ:  # valid-only needs to be set to True
        import torch
        from wisp.renderer.core import RendererCore
        import wisp.renderer.core.renderers
        from wisp.renderer.core.api import add_to_scene_graph
        from wisp.ops.image import write_png, write_exr

        from wisp.config_parser import get_modules_from_config, get_optimizer_from_config
        pipeline, train_dataset, device = get_modules_from_config(args)
        optim_cls, optim_params = get_optimizer_from_config(args)
        add_to_scene_graph(scene_state, 'scene1', pipeline) # Add pipeline to scene graph (this is also done when initializing a trainer in base_trainer.py)

        ### Second object
        fox = True
        if not fox:
            args.config = "configs/nglod_nerf.yaml"
            args.dataset_path = "/home/salar/datasets/V8_"
            args.multiview_dataset_format = "rtmv"
            args.mip = 2
            args.pretrained = "/home/salar/RVL/repos/kaolin-wisp/_results/logs/runs/test-nglod-nerf/20220822-225049/model.pth"
        else:
            args.pretrained = "/home/salar/RVL/repos/kaolin-wisp/_results/logs/runs/test-ngp-nerf/20220822-231128/model.pth"
            args.dataset_path = "../instant-ngp/data/nerf/fox/"
        pipeline2, train_dataset2, device = get_modules_from_config(args)
        add_to_scene_graph(scene_state, "object2", pipeline2)

        # ### RendererCore

        # height, width = 360, 640
        # core = RendererCore(scene_state)
        # core.resize_canvas(height=height, width=width)
        # core.set_full_resolution()
        # core.redraw()          # Call this every time you add / remove an object

        # ## Scene only _______________________________

        # ### Set camera

        # # TODO: set separate cameras for the two objects
        # #camera = train_dataset2.data.get("cameras", dict()) ['00001'] # nglod v8
        # #camera = train_dataset2.data.get("cameras", dict()) ['0001'] # ngp fox
        # val_data = train_dataset.get_images(split="val", mip=2)
        # #camera = train_dataset.data.get("cameras", dict()) ['0_00050'] # carla
        # camera = val_data.get("cameras", dict()) ['1_00001'] # carla val
        # scene_state.renderer.selected_camera = camera
        # #print(camera)

        # ### Set visible objects and render
        # scene_state.graph.visible_objects['object2'] = False
        # print(scene_state.graph.visible_objects)

        # ### Render
        # rb = core.render()     # Obtain a render buffer
        # #exrdict = rb.cpu().exr_dict()
        # #write_exr("tmp.exr", exrdict)
        # img_out = rb.cpu().image().byte().rgb.numpy()
        # print(img_out.shape)
        # write_png("tmp1.png", img_out)

        # ## Object only _______________________________
        
        # ### Set camera
        # camera = train_dataset2.data.get("cameras", dict()) ['00001'] # nglod v8
        # scene_state.renderer.selected_camera = camera
        # #print(camera)

        # ### Set visible objects and render
        # scene_state.graph.visible_objects['object2'] = True
        # scene_state.graph.visible_objects['scene1'] = False
        # print(scene_state.graph.visible_objects)

        # ### Render
        # #core.resize_canvas(height=height, width=width)
        # rb = core.render()     # Obtain a render buffer
        # img_out = rb.cpu().image().byte().rgb.numpy()
        # print(img_out.shape)
        # write_png("tmp2.png", img_out)

        ## Both Scene and Object _______________________________
        extra_args=vars(args)
        from wisp.offline_renderer import OfflineRenderer
        from wisp.core import Rays
        from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
        renderer = OfflineRenderer(**extra_args)

        data = train_dataset.get_images(split="val", mip=0)
        # rays = data["rays"]
        # imgs = list(data["imgs"])
        # img_shape = imgs[0].shape
        # ray_os = list(rays.origins)
        # ray_ds = list(rays.dirs)
        # rays = Rays(ray_os[0], ray_ds[0], dist_min=rays.dist_min, dist_max=rays.dist_max)
        # rays = rays.reshape(-1, 3)
        # rays = rays.to('cuda')
        #data = train_dataset.data
        cameras = data["cameras"]
        camera = cameras['1_00000']
        #camera.intrinsics.zoom(-30)
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device='cuda')
        rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).to(dtype=torch.float)
        rays = rays.reshape(-1, 3).to('cuda')
        img_shape = train_dataset.img_shape
        rb = renderer.render(pipeline, rays)
        rb = rb.reshape(*img_shape[:2], -1)
        img_out1 = rb.cpu().image().byte().rgb.numpy()
        depth_out1 = rb.cpu().image().byte().depth.numpy()
        write_png("tmp3.png", img_out1)
        write_png("tmp3_depth.png", depth_out1)
        print(img_out1.shape)

        #data = train_dataset2.get_images(split="val", mip=0)
        data = train_dataset2.data
        cameras = data["cameras"]
        if fox:
            camera = cameras['0001']
            camera.intrinsics.zoom(20)
        else:
            camera = cameras['00001']
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device='cuda')
        rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).to(dtype=torch.float)
        rays = rays.reshape(-1, 3).to('cuda')
        #imgs = list(data["imgs"])
        #img_shape = imgs[0].shape
        img_shape2 = train_dataset2.img_shape
        rb2 = renderer.render(pipeline2, rays)
        rb2 = rb2.reshape(*img_shape2[:2], -1)
        rb2 = rb2.scale(img_shape[:2])
        img_out2 = rb2.cpu().image().byte().rgb.numpy()
        depth_out2 = rb2.cpu().image().byte().depth.numpy()
        write_png("tmp4.png", img_out2)
        write_png("tmp4_depth.png", depth_out2)
        print(img_out2.shape)

        ## Blend
        core = RendererCore(scene_state)
        out_rb = core._create_empty_rb(height=img_shape[0], width=img_shape[1]).to('cuda')
        #print(scene_state.graph.channels.keys())
        #print(scene_state.graph.channels)
        #from wisp.core.channel_fn import *
        #scene_state.graph.channels['rgb'].blend_fn = blend_normal
        out_rb = out_rb.blend(rb, channel_kit=scene_state.graph.channels)
        out_rb = out_rb.blend(rb2, channel_kit=scene_state.graph.channels)
        img_out3 = out_rb.cpu().image().byte().rgb.numpy()
        depth_out3 = out_rb.cpu().image().byte().depth.numpy()
        write_png("tmp5.png", img_out3)
        write_png("tmp5_depth.png", depth_out3)

        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(23, 6))
        plt.subplot(2,3,1)
        plt.imshow(img_out1)
        plt.subplot(2,3,2)
        plt.imshow(img_out2)
        plt.subplot(2,3,3)
        plt.imshow(img_out3)
        plt.subplot(2,3,4)
        plt.imshow(depth_out1)
        plt.subplot(2,3,5)
        plt.imshow(depth_out2)
        plt.subplot(2,3,6)
        plt.imshow(depth_out3)
        f.savefig("tmp_compose.png", bbox_inches='tight')

        exit()

    # Create the trainer
    trainer = create_trainer(args, scene_state)

    if not os.environ.get('WISP_HEADLESS') == '1':
        interactive_app = create_app(scene_state, trainer)
        interactive_app.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()
