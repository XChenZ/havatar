import argparse
import os
import time
import cv2
from dataloader.dataloaderSR import Loader
from model.nerf_trainer import Trainer as nerf_trainer
import numpy as np
import torch
from tqdm import tqdm
import yaml
import datetime
from utils.cfgnode import CfgNode
from utils.training_util import load_partial_state_dict
from model.styleUnet import SWGAN_unet


class styleUnet_args(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ## Training
        # n_sample = 64   # number of the samples generated during training
        self.latent = 64
        self.n_mlp = 4
        self.channel_multiplier = 2  # channel multiplier factor for the model. config-f = 2, else = 1
        self.start_iter = 0
        self.batch = 2  # batch sizes for each gpus   # <=4 or 4 * N
        self.wandb = True  # use weights and biases logging
        self.lr = 0.0005  # learning rate

        self.mixing = 0.9  # probability of latent code mixing

        self.augment = True  # apply non leaking augmentation
        self.augment_p = 0.  # probability of applying augmentation. 0 = use adaptive augmentation
        self.ada_target = 0.6  # target augmentation probability for adaptive augmentation
        self.ada_length = 500 * 1000  # target duraing to reach augmentation probability for adaptive augmentation
        self.ada_every = 256  # probability update interval of the adaptive augmentation

        self.path_regularize = 2.  # weight of the path length regularization
        self.path_batch_shrink = 2  # batch size reducing factor for the path length regularization (reduce memory consumption)
        self.g_reg_every = 4  # interval of the applying path length regularization
        self.view_dis_every = 0  ############################################ zxc: round view Dis
        self.r1 = 10.  # weight of the r1 regularization
        self.d_reg_every = 16  # interval of the applying r1 regularization

su_args = styleUnet_args()

# cond_SR = False
# use_style = True
# use_noise = True
# use_SR = True
# render_nerf = False

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        if not require_all:
            assert (name in src_tensors) or (not require_all)
            try:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
            except:
                # if print_:
                #     print(name, tensor.shape, src_tensors[name].shape if name in src_tensors.keys() else '')
                # else:
                #     if name.split('.')[0] == 'backbone' or name.split('.')[0] == 'superresolution' or name.split('.')[0] == 'decoder':
                #         print('Miss, ', name)
                continue
        else:
            if name in src_tensors:
                try:
                    tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                except:
                    print(name, src_tensors[name].shape, tensor.shape)
                    assert False
            else:
                print('NotIn src_module', name)
                # print(src_tensors.keys())
                # exit(0)
                assert False

def get_center_pidx(s):
    if s > 1:
        return([(s ** 2 - s - 2) // 2, (s ** 2 - s) // 2, (s ** 2 + s - 2) // 2, (s ** 2 + s) // 2])
    else:
        return([0])


def make_noise(n_uplayers, device):
    noises = []
    # for i in range(4, self.log_size + 1):
    for i in n_uplayers:
        for _ in range(2):
            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    return noises


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='config/singleview_512_HD_base.yml', help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--savedir", type=str, default='./renders/', help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--split", type=str, default=None, help="Save images to this directory, if specified."
    )
    configargs = parser.parse_args()
    os.makedirs(os.path.join(configargs.savedir, "rgb"), exist_ok=True)
    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    render_size, gen_size = cfg.models.StyleUnet.inp_size, cfg.models.StyleUnet.out_size

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")

    print(configargs.ckpt)
    checkpoint = torch.load(configargs.ckpt)

    nerf_render = nerf_trainer(cfg, 0).requires_grad_(False).to(device)
    img_trans = SWGAN_unet(inp_size=render_size, inp_ch=cfg.models.StyleUnet.inp_ch, out_size=gen_size, out_ch=3, style_dim=su_args.latent, c_dim=0,
                           n_mlp=su_args.n_mlp, channel_multiplier=su_args.channel_multiplier).to(device)
    load_partial_state_dict(nerf_render, checkpoint["nerf_render"], except_keys=['latent_codes'])
    nerf_render.latent_codes = checkpoint['latent_codes'].to(device)
    img_trans.load_state_dict(checkpoint["g_ema"])
    nerf_render.headpose_skin_net.fix_canonical_W()
    nerf_render.eval()
    img_trans.eval()

    style = torch.mean(torch.randn(1000, 1, su_args.latent), dim=0).to(device)
    val_loader = Loader(split_file=configargs.split, mode='test', batch_size=1, options=cfg, down_sample=cfg.dataset.down_sample)

    with torch.no_grad():
        for idx, val_batch in tqdm(val_loader):
            name = str(int(val_batch['fidx'][0].numpy()))
            k = int(val_batch['vidx'][0].numpy())

            inp_data = {'mode': 'validation', 'fidx': idx, 'render_full_img': True,
                        'ray_batch': val_batch['mv_rays'][..., :-3].to(device),  # [B, 128**2, 8-1]
                        'background_prior': val_batch['mv_rays'][..., -3:].to(device),  # [B, 128**2, 3]
                        }
            inp_data.update({'front_render_cond': val_batch['front_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                             'left_render_cond': val_batch['left_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                             'right_render_cond': val_batch['right_render_cond'].permute(0, 3, 1, 2).to(device),
                             'inv_head_T': val_batch['inv_head_T'].to(device)})

            render, _, _ = nerf_render(**inp_data)  # [B, C, 128, 128]

            gen_img = img_trans(styles=[style], condition_img=render[:, 3:])
            gen_img = np.clip((gen_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0]) * 255, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(configargs.savedir, 'rgb', f"{name}_{k:02d}.png"), cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))

    print("Done!")


if __name__ == "__main__":
    main()