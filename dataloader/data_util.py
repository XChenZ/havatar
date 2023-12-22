import numpy as np
import torch


def make_ray_importance_sampling_map(mask, p=0.9):
    probs = np.zeros_like(mask).astype(np.float32)
    probs.fill(1 - p)
    probs[mask>0] = p
    probs = (1 / probs.sum()) * probs
    return probs


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def get_rays(H, W, intr, c2w, normalize=True):  ##zxc
    '''
    :param H:
    :param W:
    :param intrinsic: tensor [3, 3]
    :param c2w: tensor [3, 4]
    :param mode:
    :return: rays_o [3], rays_d [H, W, 3]
    '''

    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intr[0], intr[1], intr[2] * W, intr[3] * H
    K_inv = torch.from_numpy(np.linalg.inv(K)).to(c2w.device)
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=c2w.device),
                          torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    homo_indices = torch.stack((i, j, torch.ones_like(i)), -1)  # [H, W, 3]
    dirs = (K_inv[None, ...] @ homo_indices[..., None])[:, :, :, 0]

    # Rotate ray directions from camera frame to the world frame
    rays_d = (c2w[None, :3, :3] @ dirs[..., None])[:, :, :, 0]  # [H, W, 3]
    if normalize:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1]
    rays_o = rays_o.expand(rays_d.shape)
    return rays_o, rays_d


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [batch_size, 3] tensor containing X, Y, and Z angles.
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [batch_size, 3, 3]. rotation matrices.
    '''
    angles = angles*(np.pi)/180.
    s = torch.sin(angles)
    c = torch.cos(angles)

    cx, cy, cz = (c[:, 0], c[:, 1], c[:, 2])
    sx, sy, sz = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0]).to(angles.device)
    ones = torch.ones_like(s[:, 0]).to(angles.device)

    # Rz.dot(Ry.dot(Rx))
    R_flattened = torch.stack(
    [
      cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
          -sy,                cy * sx,                cy * cx,
    ],
    dim=0) #[batch_size, 9]
    R = torch.reshape(R_flattened, (-1, 3, 3)) #[batch_size, 3, 3]
    return R