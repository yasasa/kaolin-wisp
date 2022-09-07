from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from wisp.trainers import *
from wisp.framework import WispState
from wisp.config_parser import *

torch.manual_seed(123)

res = 16

parser = parse_options(return_parser=True)
args, args_str = argparse_to_str(parser)
nef, tracer, pipeline = get_model_from_config(args, res)
p = np.pi / 6
coords = torch.tensor([np.pi / 12, np.pi / 12, -0.2]).view(1, 1, 3)
dir = torch.randn(1, 3).cuda().requires_grad_(True)
coords = coords.cuda().requires_grad_(True)
data = torch.ones_like(nef.grid.codebook[0].data)
rand_data = torch.randn_like(nef.grid.codebook[0].data) * 0.1
data[4:] *= 2.

nef.grid.codebook[0].data = rand_data
value = nef.grid.interpolate(coords, lod_idx=0)
value.sum().backward()


def f(x):
    return nef.grid.interpolate(x, lod_idx=0).sum(dim=-1).squeeze()


@torch.no_grad()
def df_fd(f, x, eps=1e-4):
    dg = torch.zeros_like(x)
    N = x.shape[-1]
    for i in range(N):
        e_i = torch.eye(N)[i].view(1, 1, N).expand(x.shape[0], 1, N).cuda()

        x1 = x + e_i * eps
        x2 = x - e_i * eps
        dg[:, 0, i] = (f(x1) - f(x2)) / (2 * eps)

    return dg


@torch.enable_grad()
def df(f, x):
    x_ = x.detach().clone().requires_grad_(True)
    y = f(x_)
    y.backward()
    return x_.grad


sample_points = (torch.randn(1000, 1, 3) * 2 - 1).clip(
    -1 + np.pi / 64, (res - 2) / res - np.pi / 64).cuda().requires_grad_(True)

xs = np.linspace(-1, 1 - 1. / res, res, endpoint=False)[:-1] + 1. / res
grid = np.stack(np.meshgrid(xs, xs, xs)).reshape(3, -1).T
grid_points = torch.from_numpy(grid).float()
sample_points = grid_points.view(-1, 1, 3).cuda()

torch.autograd.gradcheck(f,
                         sample_points.view(-1, 1, 3).requires_grad_(True),
                         eps=1e-3,
                         atol=1e-1)
import matplotlib.pyplot as plt

fds = torch.stack([
    df_fd(f, sample_points[i].view(1, 1, 3), eps=1e-3)
    for i in range(sample_points.shape[0])
])
dfs = torch.stack([
    df(f, sample_points[i].view(1, 1, 3))
    for i in range(sample_points.shape[0])
])

diffs = (fds.squeeze() - dfs.squeeze()).norm(dim=-1)
max_diff = diffs.argmax()
if diffs.ndim > 0:
    print(diffs[max_diff], sample_points[max_diff], dfs[max_diff],
          fds[max_diff], max_diff)
else:
    print(diffs, sample_points, dfs, fds)

plt.scatter(sample_points[:, 0, 0].detach().cpu(),
            sample_points[:, 0, 1].detach().cpu(),
            c=diffs.cpu(),
            cmap='viridis')
ticks = np.linspace(-1, 1, res)
plt.xlim(-1, 1)
plt.xticks(ticks, size='xx-small')
plt.ylim(-1, 1)
plt.yticks(ticks, size='xx-small')
plt.title("Gradient error of points projected to x-y axis")
plt.grid()
plt.colorbar()
plt.savefig("x-y-scatter", dpi=300)
