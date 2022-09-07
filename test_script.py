from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from wisp.trainers import *
from wisp.framework import WispState
from wisp.config_parser import *

torch.manual_seed(123)

parser = parse_options(return_parser=True)
args, args_str = argparse_to_str(parser)
nef, tracer, pipeline = get_model_from_config(args)
p = np.pi / 6
coords = torch.tensor([np.pi / 12, np.pi / 12, -0.2]).view(1, 1, 3)
dir = torch.randn(1, 3).cuda().requires_grad_(True)
coords = coords.cuda().requires_grad_(True)
data = torch.ones_like(nef.grid.codebook[0].data)
rand_data = torch.randn_like(nef.grid.codebook[0].data)
data[4:] *= 2.
print(data)
nef.grid.codebook[0].data = rand_data
#print(nef.grid.codebook[0].data)
value = nef.grid.interpolate(coords, lod_idx=0)
value.sum().backward()
#print(value, coords.grad, dir.grad)


def f(x):
    return nef.grid.interpolate(x, lod_idx=0).sum()


def get_grad(f, x):
    x_ = x.clone().requires_grad_(True)
    y = f(x_)
    y.backward()
    return x_.grad


torch.autograd.gradcheck(f, coords, eps=1e-2, atol=1e-2)

import matplotlib.pyplot as plt

ps = torch.linspace(-1, 1., 10)
coords = torch.zeros(10, 1, 3).cuda()
coords[:, 0, 2] = ps
y = get_grad(f, coords)
plt.plot()
