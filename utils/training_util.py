import torch
import numpy as np
import math
from typing import Optional
import torchvision
import matplotlib.pyplot as plt

def merge_continuous_dimensions(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    if x.is_contiguous():
        return x.view(combined_shape)
    else:
        return x.reshape(combined_shape)


def img2mse(img_src, img_tgt, loss_flag='mse'):
    assert loss_flag in ['mse', 'l1']
    if loss_flag == 'mse':
        return torch.nn.functional.mse_loss(img_src, img_tgt)
    else:
        return torch.nn.functional.l1_loss(img_src, img_tgt)


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8, dim=0):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    if dim == 0:
        return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    else:
        return [inputs[:, i: i + chunksize] for i in range(0, inputs.shape[1], chunksize)]


def cast_to_image(tensor, dataformats='CHW'):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    if dataformats=='CHW':
        img = np.moveaxis(img, [-1], [0])
    return img


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def torch_normal_map(depthmap, intr, weights=None, clean=True, central_difference=False):
    from dataloader.data_util import meshgrid_xy

    W, H = depthmap.shape
    #normals = torch.zeros((H,W,3), device=depthmap.device)
    fx, fy, cx, cy = intr[0], intr[1], intr[2] * W, intr[3] * H
    ii, jj = meshgrid_xy(torch.arange(W, device=depthmap.device),
                         torch.arange(H, device=depthmap.device))
    points = torch.stack(
        [
            ((ii - cx) * depthmap) / fx,
            -((jj - cy) * depthmap) / fy,
            depthmap,
        ],
        dim=-1)
    difference = 2 if central_difference else 1
    dx = (points[difference:,:,:] - points[:-difference,:,:])
    dy = (points[:,difference:,:] - points[:,:-difference,:])
    normals = torch.cross(dy[:-difference,:,:],dx[:,:-difference,:],2)
    normalize_factor = torch.sqrt(torch.sum(normals*normals,2))
    normals[:,:,0]  /= normalize_factor
    normals[:,:,1]  /= normalize_factor
    normals[:,:,2]  /= normalize_factor
    normals = normals * 0.5 +0.5

    if clean and weights is not None: # Use volumetric rendering weights to clean up the normal map
        mask = weights.repeat(3,1,1).permute(1,2,0)
        mask = mask[:-difference,:-difference]
        where = torch.where(mask > 0.22)
        normals[where] = 1.0
        normals = (1-mask)*normals + (mask)*torch.ones_like(normals)
    normals *= 255
    #plt.imshow(normals.cpu().numpy().astype('uint8'))
    #plt.show()
    return normals


def save_plt_image(im1, outname):
    fig = plt.figure()
    fig.set_size_inches((6.4,6.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plt.set_cmap('jet')
    ax.imshow(im1, aspect='equal')
    plt.savefig(outname, dpi=80)
    plt.close(fig)


def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


def lpips_loss(img0, img1, lpips_fn):
    # img: [B, 3, H, W], tensor, 0, 1
    img0 = (img0 * 2.) - 1.0
    img1 = (img1 * 2.) - 1.0
    loss = lpips_fn.forward(img0, img1)

    # print(type(loss), loss.shape)
    return loss.mean()


def load_partial_state_dict(model, loaded_dict, except_keys, full_name=False):
    model_dict = model.state_dict()
    if full_name:
        pretrained_dict = {k: v for k, v in loaded_dict.items() if k not in except_keys}
    else:
        pretrained_dict = {}
        for k, v in loaded_dict.items():
            flag = False
            for except_key in except_keys:
                if k.startswith(except_key):
                    flag = True
                    break
            if not flag:
                pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def create_code_snapshot(root, dst_path, extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(
                    root).as_posix(), recursive=True)