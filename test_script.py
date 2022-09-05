from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from wisp.trainers import *
from wisp.framework import WispState
from wisp.config_parser import *

parser = parse_options(return_parser=True)
args, args_str = argparse_to_str(parser)
nef, tracer, pipeline = get_model_from_config(args)

coords = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3)
dir = torch.randn(1, 3).cuda().requires_grad_(True)
coords = coords.cuda().requires_grad_(True)
data = torch.ones_like(nef.grid.codebook[0].data)
data[4:] *= 2.
nef.grid.codebook[0].data = data
print(nef.grid.codebook[0].data)
value = nef.rgba(coords, dir, lod_idx=0)['rgb']
value.sum().backward()
print(value, coords.grad, dir.grad)


def f(x):
    return nef.rgba(x, dir)['rgb'].sum()


torch.autograd.gradcheck(f, coords, eps=1e-2, atol=1e-2)
