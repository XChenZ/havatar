
import argparse
import os
import time
import sys
from dataloader.dataloaderSR import Loader

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG'] = '3'
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
# from model_for_onnx.nerf_trainer_clean import Trainer
# from model_for_onnx.styleUnet import SWGAN_unet
from model.nerf_trainer import Trainer
from model.styleUnet import SWGAN_unet
import torch.nn.functional as F
from utils.training_util import mse2psnr, lpips_loss, load_partial_state_dict
from utils.cfgnode import CfgNode

import lpips
from utils.styleUnet_util import sample_data, mixing_noise, requires_grad, g_nonsaturating_loss, g_path_regularize, d_logistic_loss, accumulate, d_r1_loss, styleUnet_args
from model.styleUnet import Discriminator
from dataloader.dist_util import synchronize, get_rank

su_args = styleUnet_args()
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
                assert False

def create_code_snapshot(root, dst_path,
                         extensions=(".py", ".h", ".cpp", ".cu",
                                     ".cc", ".cuh", ".json", ".sh", ".bat"),
                         exclude=()):
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


def main():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--config", type=str, default='config/singleview_512_HD_base.yml', help="Path to (.yml) config file.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to load saved checkpoint from.")
    parser.add_argument("--continue-training", action='store_true', default=False)
    configargs = parser.parse_args()

    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    render_size, gen_size = cfg.models.StyleUnet.inp_size, cfg.models.StyleUnet.out_size

    device = torch.device("cuda")
    lpips_fn = lpips.LPIPS().to(device)
    use_percep_loss = True
    percep_loss_fn = None
    if use_percep_loss:
        percep_loss_fn = lpips.LPIPS(net='vgg').to(device)

    # Load dataset
    train_loader = Loader(split_file=os.path.join(configargs.datadir, 'mv_v31_all.json'), down_sample=cfg.dataset.down_sample,
                          mode='train', batch_size=su_args.batch, options=cfg, white_bg=True)

    ##################################
    # ---------------------------------- Create Nerf Trainner & Discriminator ----------------------------------
    nerf_render = Trainer(cfg, len(train_loader.dataset)).to(device)
    generator = SWGAN_unet(inp_size=render_size, inp_ch=cfg.models.StyleUnet.inp_ch, out_size=gen_size, out_ch=3, style_dim=su_args.latent, c_dim=0,
                           n_mlp=su_args.n_mlp, channel_multiplier=su_args.channel_multiplier).to(device)
    discriminator = Discriminator(gen_size, 3, channel_multiplier=su_args.channel_multiplier, c_dim=0).to(device)
    g_ema = SWGAN_unet(inp_size=render_size, inp_ch=cfg.models.StyleUnet.inp_ch, out_size=gen_size, out_ch=3, style_dim=su_args.latent, c_dim=0,
                       n_mlp=su_args.n_mlp, channel_multiplier=su_args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_reg_ratio = su_args.g_reg_every / (su_args.g_reg_every + 1)
    d_reg_ratio = su_args.d_reg_every / (su_args.d_reg_every + 1)
    su_args.lr = 1e-3
    g_optim = torch.optim.Adam(generator.parameters(), lr=su_args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=su_args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),)
    nerf_optimizer = getattr(torch.optim, cfg.optimizer.type)([{'params': nerf_render.parameters()}], lr=cfg.optimizer.lr)

    ##################################
    # ---------------------------------- Setup Logging & Load Checkpoint ----------------------------------
    logdir = configargs.logdir
    os.makedirs(logdir, exist_ok=True)
    save_dir = os.path.join(logdir, 'sample')

    # Write out config parameters.
    if get_rank() == 0:
        with open(os.path.join(logdir, "config.yml"), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    start_iter = -1
    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.ckpt):
        print(configargs.ckpt)
        if not configargs.continue_training:   # load from train_avatar_mp_multiRender
            checkpoint = torch.load(configargs.ckpt)  # , map_location=lambda storage, loc: storage)
            nerf_render.load_state_dict(checkpoint["trainer_state_dict"])

            # load pretrained gan
            checkpoint = torch.load('pretrained_models/img_translation.ckpt')
            generator.load_state_dict(checkpoint["g"])
            discriminator.load_state_dict(checkpoint["d"])
            g_ema.load_state_dict(checkpoint["g_ema"])
            del checkpoint
        else:   # continue training
            checkpoint = torch.load(configargs.ckpt)#, map_location=lambda storage, loc: storage)
            nerf_render.load_state_dict(checkpoint["nerf_render"])
            nerf_optimizer.load_state_dict(checkpoint["nerf_optimizer"])
            generator.load_state_dict(checkpoint["g"])
            discriminator.load_state_dict(checkpoint["d"])
            g_ema.load_state_dict(checkpoint["g_ema"])
            g_optim.load_state_dict(checkpoint["g_optim"])
            d_optim.load_state_dict(checkpoint["d_optim"])
            # start_iter = checkpoint["iter"]
            del checkpoint

    sample_z = torch.randn(su_args.batch, su_args.latent, device=device)
    accum = 0.5 ** (32 / (10 * 1000))

    rgb_loss_func = torch.nn.functional.mse_loss if cfg.experiment.rgb_loss == 'mse' else torch.nn.functional.l1_loss
    # i = start_iter
    loss_dict = {}

    g_module = generator
    d_module = discriminator
    nerf_moudule = nerf_render

    path_loss, path_lengths, r1_loss, mean_path_length = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), 0
    pbar = tqdm(range(su_args.iter), initial=su_args.start_iter, dynamic_ncols=True, smoothing=0.01)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'checkpoint'), exist_ok=True)
    tar_file = os.path.join(logdir, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    create_code_snapshot(os.path.split(os.path.abspath(__file__))[0], tar_file)
    writer = SummaryWriter(logdir)
    loader = sample_data(train_loader)

    for idx in pbar:
        i = idx + start_iter + 1
        if i > su_args.iter:
            print("Done!")
            break
        fidx, train_batch = next(loader)
        batch_num = len(fidx)
        gt_hr_img = train_batch['mv_rays_gt_color'].permute(0, 2, 1).reshape(batch_num, 3, gen_size, gen_size).to(device)   # [B, 3, 512**2]
        gt_lr_mask = train_batch['mv_rays'][..., -1:].permute(0, 2, 1).reshape(batch_num, 1, render_size, render_size).to(device)    # [B, 1, 128**2]

        inp_data = {'mode':'train', 'fidx': fidx, 'render_full_img':True,
                    'ray_batch': train_batch['mv_rays'][..., :-4].to(device),  # [B, 128**2, 8-1]
                    'background_prior': train_batch['mv_rays'][..., -4:-1].to(device),  # [B, 128**2, 3]
                    }
        inp_data.update({'front_render_cond': train_batch['front_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                         'left_render_cond': train_batch['left_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                         'right_render_cond': train_batch['right_render_cond'].permute(0, 3, 1, 2).to(device),
                         'inv_head_T': train_batch['inv_head_T'].to(device)})

        ##################################
        # ---------------------------------- Nerf render LR ----------------------------------
        gt_lr_img = torch.nn.functional.interpolate(
            torch.nn.functional.interpolate(gt_hr_img, size=(render_size, render_size), mode='bilinear', align_corners=True),
            size=(gen_size, gen_size), mode='bilinear', align_corners=True)
        gan_weight = 1.1 ** (i // 500)
        gan_loss_weight = min(1e-3 * gan_weight, 0.1)

        # ---------------------------------- StyleUnet generate HR ----------------------------------
        g_regularize = i % su_args.g_reg_every == 0  # path length regularization
        d_regularize = i % su_args.d_reg_every == 0
        requires_grad(nerf_render, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        with torch.no_grad():
            render, _, _ = nerf_render(**inp_data)  # [B, C, 128, 128]
            noise = mixing_noise(batch_num, su_args.latent, su_args.mixing, device)
            fake_img = generator(noise, render[:, 3:])
            # lr_img = torch.nn.functional.interpolate(render[:, :3], size=(gen_size, gen_size), mode='bilinear', align_corners=True)
        fake_pred = discriminator(fake_img, flat_pose=None)
        real_pred = discriminator(gt_hr_img, flat_pose=None)
        # fake_pred = discriminator(torch.cat([fake_img, lr_img], dim=1))
        # real_pred = discriminator(torch.cat([gt_hr_img, gt_lr_img], dim=1))
        d_loss = d_logistic_loss(real_pred, fake_pred) * gan_loss_weight

        loss_dict["d"] = d_loss / gan_loss_weight
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if d_regularize:
            gt_hr_img.requires_grad = True
            # real_pred = discriminator(torch.cat([gt_hr_img, gt_lr_img], dim=1))
            real_pred = discriminator(gt_hr_img, flat_pose=None)
            r1_loss = d_r1_loss(real_pred, gt_hr_img) * gan_loss_weight
            discriminator.zero_grad()
            (su_args.r1 / 2 * r1_loss * su_args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()
        loss_dict["r1"] = r1_loss / gan_loss_weight


        requires_grad(nerf_render, True)
        render, mask, latent_code_loss = nerf_render(**inp_data)  # [B, C, 128, 128]
        lr_img = torch.nn.functional.interpolate(render[:, :3], size=(gen_size, gen_size), mode='bilinear', align_corners=True)
        rgb_loss = rgb_loss_func(lr_img, gt_lr_img)
        loss_dict["rgb_loss"] = rgb_loss.item()
        nerf_loss = (rgb_loss + 1. * latent_code_loss)
        if cfg.experiment.mask_weight > 0:
            mask_loss = cfg.experiment.mask_weight * F.binary_cross_entropy(mask.clip(1e-3, 1.0 - 1e-3), gt_lr_mask)
            loss_dict["mask_loss"] = mask_loss.item()
            nerf_loss += mask_loss
        loss_dict["nerf_loss"] = nerf_loss.item()

        g_loss = nerf_loss
        requires_grad(generator, True)

        noise = mixing_noise(batch_num, su_args.latent, su_args.mixing, device)
        fake_img = generator(noise, render[:, 3:])

        requires_grad(discriminator, False)
        # fake_pred = discriminator(torch.cat([fake_img, lr_img], dim=1))
        fake_pred = discriminator(fake_img, flat_pose=None)
        g_loss += g_nonsaturating_loss(fake_pred) * gan_loss_weight
        loss_dict["g"] = g_nonsaturating_loss(fake_pred)

        hr_l1_loss = torch.nn.functional.l1_loss(fake_img, gt_hr_img)
        g_loss += hr_l1_loss #* 0
        loss_dict["hr_l1"] = hr_l1_loss

        if use_percep_loss:
            percep_loss = lpips_loss(fake_img, gt_hr_img, percep_loss_fn)
            g_loss += percep_loss * 0.1

        nerf_render.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        nerf_optimizer.step()

        psnr = mse2psnr(torch.nn.functional.mse_loss(lr_img, gt_lr_img).item())
        SR_psnr = mse2psnr(torch.nn.functional.mse_loss(fake_img, gt_hr_img).item())

        if False:#use_style and g_regularize: #TODO：derivative for cudnn_grid_sampler_backward is not implemented
            path_batch_size = max(1, batch_num // su_args.path_batch_shrink)
            requires_grad(nerf_render, False)
            path_cond_img = render[:path_batch_size, 3:].detach().clone()
            path_cond_img.requires_grad = True
            # render, _ = nerf_render(inp_data)  # [B, C, 128, 128]
            noise = mixing_noise(path_batch_size, su_args.latent, su_args.mixing, device)
            fake_img, latents = generator(noise, path_cond_img, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)

            generator.zero_grad()
            weighted_path_loss = su_args.path_regularize * su_args.g_reg_every * path_loss
            if su_args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
            weighted_path_loss.backward()
            g_optim.step()

        accumulate(g_ema, g_module, accum)

        hr_l1 = loss_dict["hr_l1"].item()
        d_loss_val = loss_dict["d"].mean().item()
        g_loss_val = loss_dict["g"].mean().item()
        r1_val = loss_dict["r1"].mean().item()
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"PSNR: {psnr:.4f};"
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    # f"path: {path_loss_val:.4f}; mean path: {mean_path_length.item():.4f}; "
                )
            )
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
            writer.add_scalar("train/rgb_loss", loss_dict["rgb_loss"], i)
            if cfg.experiment.mask_weight > 0:
                writer.add_scalar("train/mask_loss", loss_dict["mask_loss"], i)

            writer.add_scalar("train/psnr", psnr, i)
            writer.add_scalar("train/d_loss_val", d_loss_val, i)
            writer.add_scalar("train/g_loss_val", g_loss_val, i)
            writer.add_scalar("train/r1_val", r1_val, i)
            writer.add_scalar("train/SR_psnr", SR_psnr, i)
            writer.add_scalar("train/SR_l1", hr_l1, i)
            if percep_loss_fn is not None:
                writer.add_scalar("train/percep_loss", percep_loss.item(), i)

        if get_rank() == 0:
            if i % cfg.experiment.validate_every == 0 or i == start_iter + 1:
                with torch.no_grad():
                    g_ema.eval()
                    noise = [sample_z[:batch_num]]
                    sample = g_ema(noise, render[:, 3:])
                    lpips_value = lpips_loss(sample, gt_hr_img, lpips_fn)
                    writer.add_scalar("train_val/lpips", lpips_value, i)
                    torchvision.utils.save_image(
                        tensor=torch.cat([sample, lr_img, gt_hr_img], dim=3),
                        fp=f"{save_dir}/{str(i).zfill(6)}.png",
                        nrow=int(batch_num ** 0.5),
                        normalize=True,
                        range=(0, 1),
                    )

                checkpoint_dict = {
                    "iter": i,
                    "nerf_optimizer": nerf_optimizer.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "nerf_render": nerf_moudule.state_dict(),
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "latent_codes": nerf_moudule.latent_codes.data,
                }
                torch.save(checkpoint_dict, os.path.join(logdir, "latest.pt"))

            if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1 or i == start_iter + 1:
                checkpoint_dict = {
                    "iter": i,
                    "nerf_optimizer": nerf_optimizer.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "nerf_render": nerf_moudule.state_dict(),
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "latent_codes": nerf_moudule.latent_codes.data,
                }

                torch.save(
                    checkpoint_dict,
                    os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
                )
                tqdm.write("================== Saved Checkpoint =================")

    print("Done!")

def cast_to_image(tensor, dataformats='CHW'):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0, 1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    if dataformats == 'CHW':
        img = np.moveaxis(img, [-1], [0])
    return img


def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


def adjust_lr(cfg, i, optimizer):
    num_decay_steps = cfg.scheduler.lr_decay * 1000
    lr_new = cfg.optimizer.lr * (cfg.scheduler.lr_decay_factor ** (i / num_decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new
    return lr_new


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()