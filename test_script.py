from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from wisp.trainers import *
from wisp.framework import WispState
from wisp.config_parser import *

torch.manual_seed(123)

res = 6

parser = parse_options(return_parser=True)
args, args_str = argparse_to_str(parser)
nef, tracer, pipeline = get_model_from_config(args, res)
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


@torch.no_grad()
def df_fd(f, x, eps=1e-4):
    dg = torch.zeros_like(x)
    N = x.shape[-1]
    for i in range(N):
        e_i = torch.eye(x.shape[0])[i].view(1, N).expand(x.shape[0], N)
        x1 = x + e_i * eps
        x2 = x - e_i * eps
        dg[:, i] = (f(x1) - f(x2)) / (2 * eps)

    return dg


def df(f, x):
    x_ = x.clone().requires_grad_(True)
    y = f(x_)
    y.backward()
    return x_.grad


sample_points = (torch.rand(1000, 1, 3) * np.pi).clip(
    -1 + np.pi / 128,
    (res - 2) / res - np.pi / 128).cuda().requires_grad_(True)
print(sample_points[0])

torch.autograd.gradcheck(f,
                         sample_points[0].view(-1, 1, 3),
                         eps=1e-2,
                         atol=1e-2)

import matplotlib.pyplot as plt

ps = torch.linspace(-1, 1., 10)
coords = torch.zeros(10, 1, 3).cuda()
coords[:, 0, 2] = ps
y = df(f, coords)
