import argparse
import os
import time
import sys
import cv2
from dataloader.dataloader import Loader

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from utils.training_util import cast_to_image, mse2psnr, create_code_snapshot
from utils.cfgnode import CfgNode
from model_for_onnx.nerf_trainer_clean import Trainer
# from model.nerf_trainer import Trainer
import lpips


def lpips_loss(img0, img1, lpips_fn):
    # img: [B, H, W, 3], tensor, 0, 1
    img0 = (img0.permute(0, 3, 1, 2) * 2.) - 1.0
    img1 = (img1.permute(0, 3, 1, 2) * 2.) - 1.0
    loss = lpips_fn.forward(img0, img1)
    return loss.mean()

def main():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--config", type=str, default='config/singleview_512_base.yml', help="Path to (.yml) config file.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to load saved checkpoint from.")
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    device_num = torch.cuda.device_count()

    lpips_fn = lpips.LPIPS()
    percep_loss_fn = None
    if cfg.experiment.patch_rgb:
        percep_loss_fn = lpips.LPIPS(net='vgg').to(device)

    rgb_loss_func = torch.nn.functional.mse_loss if cfg.experiment.rgb_loss == 'mse' else torch.nn.functional.l1_loss
    # Load dataset
    train_loader = Loader(split_file=os.path.join(configargs.datadir, 'sv_v31_all.json'),
                          mode='train', batch_size=2, num_workers=8, down_sample=cfg.dataset.down_sample, options=cfg, white_bg=True)
    val_loader = Loader(split_file=os.path.join(configargs.datadir, 'sv_v31_all.json'), shuffle=True,
                        mode='val', batch_size=1, num_workers=1, down_sample=1.0, options=cfg, white_bg=True)
    val_data = enumerate(val_loader)
    val_img_h, val_img_w = val_loader.dataset.img_h, val_loader.dataset.img_w

    trainer = Trainer(cfg, len(train_loader.dataset)).to(device)
    # trainer = torch.nn.DataParallel(trainer.cuda())
    trainable_parameters = list(trainer.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)([{'params': trainable_parameters}], lr=cfg.optimizer.lr)

    # Setup logging.
    logdir = configargs.logdir
    os.makedirs(logdir, exist_ok=True)
    tar_file = os.path.join(logdir, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    create_code_snapshot(os.path.split(os.path.abspath(__file__))[0], tar_file)
    writer = SummaryWriter(logdir)

    # Write out config parameters.
    with open(os.path.join(logdir, 'config_%s.tar.yml' % now.strftime('%Y_%m_%d_%H_%M_%S')), "w") as f:
        f.write(cfg.dump())


    start_iter = -1
    # Load an existing checkpoint, if a path is specified.
    if len(configargs.ckpt) > 0:
        assert os.path.exists(configargs.ckpt)
        checkpoint = torch.load(configargs.ckpt)
        trainer.load_state_dict(checkpoint["trainer_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]
    else:
        if trainer.headpose_skin_net is not None:   # 初始化转脖子blending weight field
            trainer.headpose_skin_net.pretrain_wc(num_iter=3000, vol_thr=cfg.models.coarse.Head_bounding)
    if trainer.headpose_skin_net is not None:
        os.makedirs('debug', exist_ok=True)
        trainer.headpose_skin_net.visualize_motion_weight_vol('debug/vis_motionWeightVol.obj')

    i = start_iter

    while i < cfg.experiment.train_iters:
        trainer.train()

        t0 = time.time()
        for idx, train_batch in train_loader:
            i += 1
            mv_rays = train_batch['mv_rays'].to(device)
            target_ray_values = train_batch['mv_rays_gt_color'].to(device)
            ray_mask = mv_rays[..., -1:]

            inp_data = {'mode': 'train', 'fidx': idx, 'render_full_img': False,
                        'ray_batch': mv_rays[..., :-4],
                        'background_prior': mv_rays[..., -4:-1],
                        }

            inp_data.update({'front_render_cond': train_batch['front_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                             'left_render_cond': train_batch['left_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                             'right_render_cond': train_batch['right_render_cond'].permute(0, 3, 1, 2).to(device),
                             'inv_head_T': train_batch['inv_head_T'].to(device)})
            rgb_coarse, _, acc_coarse, weights, rgb_fine, _, acc_fine, latent_code_loss = trainer(**inp_data)

            # regularize_pose_skinning:
            weight_volume = trainer.headpose_skin_net.canonical_Wvolume()[0, 1]  # [32, 32, 32]
            vol_core = weight_volume[1:-1, 1:-1, 1:-1]
            vol_else = [weight_volume[:-2, 1:-1, 1:-1], weight_volume[2:, 1:-1, 1:-1], weight_volume[1:-1, 2:, 1:-1],
                        weight_volume[1:-1, :-2, 1:-1], weight_volume[1:-1, 1:-1, 2:], weight_volume[1:-1, 1:-1, :-2]]
            gradientV = sum([torch.abs(vol_core - vol) for vol in vol_else]) / 6.
            sw_grad_loss = torch.mean(gradientV)

            coarse_loss = rgb_loss_func(rgb_coarse[..., :3], target_ray_values[..., :3])
            mask_coarse_loss = F.binary_cross_entropy(acc_coarse.clip(1e-3, 1.0 - 1e-3), ray_mask)
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = rgb_loss_func(rgb_fine[..., :3], target_ray_values[..., :3])
                mask_fine_loss = F.binary_cross_entropy(acc_fine.clip(1e-3, 1.0 - 1e-3), ray_mask)

            if cfg.experiment.patch_rgb:
                patch_rgb = rgb_coarse[..., :3] if rgb_fine is None else rgb_fine[..., :3]  # [B, N, 3]
                patch_gt = target_ray_values[..., :3].reshape(patch_rgb.shape[0], int(patch_rgb.shape[1]**0.5), int(patch_rgb.shape[1]**0.5), 3)
                patch_rgb = patch_rgb.reshape(patch_rgb.shape[0], int(patch_rgb.shape[1]**0.5), int(patch_rgb.shape[1]**0.5), 3)
                patch_percep_loss = lpips_loss(patch_rgb, patch_gt, percep_loss_fn)

            loss = coarse_loss + cfg.experiment.mask_weight * mask_coarse_loss + (patch_percep_loss * 0.05 if cfg.experiment.patch_rgb else 0.0) + \
                   ((fine_loss + cfg.experiment.mask_weight * mask_fine_loss) if fine_loss is not None else 0.0)
            loss = loss + latent_code_loss + sw_grad_loss*1e-4
            psnr = mse2psnr(torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_ray_values[..., :3]).item()) if rgb_fine is None else \
                mse2psnr(torch.nn.functional.mse_loss(rgb_fine[..., :3], target_ray_values[..., :3]).item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Learning rate updates
            num_decay_steps = cfg.scheduler.lr_decay * 1000
            lr_new = max(cfg.optimizer.lr * (cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)), 5e-5)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_new

            if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
                tqdm.write(
                    "[TRAIN] Iter: " + str(i) + " Loss: %.06f" % (loss.item()) + " PSNR: %.06f" % (psnr) +
                    " LatentReg: %.04f e-5" % (1e5 * latent_code_loss.item()) + " LR: %.02f e-5" % (1e5 * lr_new) +
                    " TIME: %.02f" % ((time.time()-t0)/cfg.experiment.print_every)
                )
                t0 = time.time()

            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
            writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
            writer.add_scalar("train/mask_coarse_loss", mask_coarse_loss.item(), i)
            if rgb_fine is not None:
                writer.add_scalar("train/fine_loss", fine_loss.item(), i)
                writer.add_scalar("train/mask_fine_loss", mask_fine_loss.item(), i)

            writer.add_scalar("train/sw_grad_loss", sw_grad_loss.item(), i)
            if cfg.experiment.patch_rgb:
                writer.add_scalar("train/patch_percep_loss", patch_percep_loss.item(), i)

            writer.add_scalar("train/psnr", psnr, i)

            # Validation
            if (i==start_iter+1) or (i % cfg.experiment.validate_every == 0 and True):
                #torch.cuda.empty_cache()
                tqdm.write("[VAL] =======> Iter: " + str(i))
                trainer.eval()

                start = time.time()
                with torch.no_grad():
                    try:
                        _, val_batch = val_data.__next__()
                    except StopIteration:
                        val_data = enumerate(val_loader)
                        _, val_batch = val_data.__next__()
                    val_idx, val_batch = val_batch[0], val_batch[1]
                    val_mv_rays = val_batch['mv_rays'][0].reshape(-1, val_batch['mv_rays'][0].shape[-1])
                    val_inp_data = {'mode': 'validation', 'fidx': val_idx, 'render_full_img': False,
                                    'front_render_cond': val_batch['front_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                                    'left_render_cond': val_batch['left_render_cond'].permute(0, 3, 1, 2).to(device),  # [B, C, H, W]
                                    'right_render_cond': val_batch['right_render_cond'].permute(0, 3, 1, 2).to(device),
                                    'inv_head_T': val_batch['inv_head_T'].to(device)
                                }
                    val_view_num = int(val_mv_rays.shape[0] / (val_img_h * val_img_w))
                    rays_num, group_size = val_mv_rays.shape[0], getattr(cfg.nerf, 'validation').chunksize
                    val_rgb_coarse, val_rgb_fine, val_weights, val_acc_coarse, val_acc_fine = [], [], [], [], []
                    group_num = ((rays_num // group_size) if (rays_num % group_size == 0) else (rays_num // group_size + 1))
                    if group_num == 0:
                        group_num = 1
                        group_size = rays_num
                    for gi in range(group_num):
                        start = gi * group_size
                        end = (gi + 1) * group_size
                        end = (end if (end <= rays_num) else rays_num)
                        val_inp_data.update({
                            'ray_batch': val_mv_rays[start:end][..., :-3].to(device).unsqueeze(0),
                            'background_prior': val_mv_rays[start:end][..., -3:].to(device).unsqueeze(0),
                        })

                        rgb_coarse, _, acc_coarse, weights, rgb_fine, _, acc_fine, _ = trainer(**val_inp_data)
                        val_rgb_coarse.append(rgb_coarse[0][..., :3].detach().cpu())
                        val_weights.append(weights[0].detach().cpu())
                        val_acc_coarse.append(acc_coarse[0].detach().cpu())
                        if rgb_fine is not None:
                            val_rgb_fine.append(rgb_fine[0][..., :3].detach().cpu())
                            val_acc_fine.append(acc_fine[0].detach().cpu())
                    val_rgb_coarse = torch.cat(val_rgb_coarse, 0)
                    val_weights = torch.cat(val_weights, 0)
                    val_weights = val_weights.reshape(val_view_num, val_img_h, val_img_w)
                    val_rgb_coarse = val_rgb_coarse.reshape(val_view_num, val_img_h, val_img_w, 3)
                    val_acc_coarse = torch.cat(val_acc_coarse, 0)
                    val_acc_coarse = val_acc_coarse.reshape(val_view_num, val_img_h, val_img_w)
                    val_target_ray_values = val_batch['mv_rays_gt_color'][0].reshape(val_view_num, val_img_h, val_img_w, 3)
                    coarse_loss = rgb_loss_func(val_rgb_coarse[..., :3], val_target_ray_values[..., :3])
                    show_img_coarse = [cast_to_image(val_rgb_coarse[k, :, :, :3]) for k in range(val_view_num)]
                    show_target = [cast_to_image(val_target_ray_values[k, :, :, :3]) for k in range(val_view_num)]
                    vis_weights = [val_weights[k, :, :].numpy() for k in range(val_view_num)]
                    val_acc_coarse = [val_acc_coarse[k, :, :].numpy() for k in range(val_view_num)]
                    coarse_lpips_loss = lpips_loss(val_rgb_coarse[..., :3], val_target_ray_values[..., :3], lpips_fn)

                    if rgb_fine is not None:
                        val_rgb_fine = torch.cat(val_rgb_fine, 0)
                        val_rgb_fine = val_rgb_fine.reshape(val_view_num, val_img_h, val_img_w, 3)
                        val_acc_fine = torch.cat(val_acc_fine, 0)
                        val_acc_fine = val_acc_fine.reshape(val_view_num, val_img_h, val_img_w)
                        val_acc_fine = [val_acc_fine[k, :, :].numpy() for k in range(val_view_num)]
                        fine_loss = rgb_loss_func(val_rgb_fine[..., :3], val_target_ray_values[..., :3])
                        fine_lpips_loss = lpips_loss(val_rgb_fine[..., :3], val_target_ray_values[..., :3], lpips_fn)
                        show_img_fine = [cast_to_image(val_rgb_fine[k, :, :, :3]) for k in range(val_view_num)]
                        if len(show_img_fine) == 6:
                            vis_img_fine = np.concatenate([np.concatenate(show_img_fine[:3], axis=2), np.concatenate(show_img_fine[3:], axis=2)],
                                                          1)  # CHW
                            vis_acc_fine = np.concatenate([np.concatenate(val_acc_fine[:3], axis=1), np.concatenate(val_acc_fine[3:], axis=1)], 0)

                        else:
                            # vis_img_fine = show_img_fine[0]
                            # vis_acc_fine = val_acc_fine[0]
                            vis_img_fine = np.concatenate(show_img_fine, axis=1)
                            vis_acc_fine = np.concatenate(val_acc_fine, axis=1)

                        writer.add_image("validation/rgb_fine",
                                         cv2.resize(vis_img_fine.transpose(1, 2, 0), dsize=(0, 0), fx=1.0, fy=1.0).transpose(2, 0, 1), i)
                        writer.add_image("validation/acc_fine", cv2.resize(vis_acc_fine, dsize=(0, 0), fx=1.0, fy=1.0), i, dataformats='HW')
                        writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                        writer.add_scalar("validation/fine_lpips_loss", fine_lpips_loss.item(), i)

                    curr_fine_loss = fine_loss if rgb_fine is not None else coarse_loss
                    psnr = mse2psnr(torch.nn.functional.mse_loss(val_rgb_coarse[..., :3], val_target_ray_values[..., :3]).item()) if rgb_fine is None else \
                        mse2psnr(torch.nn.functional.mse_loss(val_rgb_fine[..., :3], val_target_ray_values[..., :3]).item())

                    writer.add_scalar("validation/psnr", psnr, i)
                    writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                    writer.add_scalar("validation/coarse_lpips_loss", coarse_lpips_loss.item(), i)

                    if len(show_img_coarse) == 6:
                        vis_img_coarse = np.concatenate([np.concatenate(show_img_coarse[:3], axis=2), np.concatenate(show_img_coarse[3:], axis=2)], 1)  # CHW
                        vis_img_target = np.concatenate([np.concatenate(show_target[:3], axis=2), np.concatenate(show_target[3:], axis=2)], 1)
                        vis_weights_map = np.concatenate([np.concatenate(vis_weights[:3], axis=1), np.concatenate(vis_weights[3:], axis=1)], 0)
                        vis_acc_coarse = np.concatenate([np.concatenate(val_acc_coarse[:3], axis=1), np.concatenate(val_acc_coarse[3:], axis=1)], 0)
                    else:
                        vis_img_coarse = np.concatenate(show_img_coarse, axis=1)
                        vis_img_target = np.concatenate(show_target, axis=1)
                        vis_weights_map = np.concatenate(vis_weights, axis=1)
                        vis_acc_coarse = np.concatenate(val_acc_coarse, axis=1)

                    writer.add_image("validation/rgb_coarse",
                                     cv2.resize(vis_img_coarse.transpose(1, 2, 0), dsize=(0, 0), fx=1.0, fy=1.0).transpose(2, 0, 1), i)
                    writer.add_image("validation/img_target",
                                     cv2.resize(vis_img_target.transpose(1, 2, 0), dsize=(0, 0), fx=1.0, fy=1.0).transpose(2, 0, 1), i)
                    writer.add_image("validation/weights", cv2.resize(vis_weights_map, dsize=(0, 0), fx=1.0, fy=1.0), i, dataformats='HW')
                    writer.add_image("validation/acc_coarse", cv2.resize(vis_acc_coarse, dsize=(0, 0), fx=1.0, fy=1.0), i, dataformats='HW')
                    err_img = np.linalg.norm((vis_img_target-vis_img_fine).transpose(1, 2, 0), axis=2) if rgb_fine is not None else \
                        np.linalg.norm((vis_img_target-vis_img_coarse).transpose(1, 2, 0), axis=2)
                    vis_err_img = cv2.applyColorMap((np.clip(err_img/300, 0., 1.) * 255).astype(np.uint8), colormap=cv2.COLORMAP_JET)
                    writer.add_image("validation/err_img",
                                     cv2.resize(cv2.cvtColor(vis_err_img, code=cv2.COLOR_BGR2RGB), dsize=(0, 0), fx=1.0, fy=1.0).transpose(2, 0, 1), i)

                    tqdm.write(
                        "Validation loss: %06f" % (curr_fine_loss.item())
                        + " Validation PSNR: %06f" % (psnr)
                        + " Time: %03f" % (time.time() - start)
                    )


            if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1 or i == start_iter + 1:
                checkpoint_dict = {
                    "iter": i,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr,
                    "trainer_state_dict": trainer.state_dict()
                }
                trainer.headpose_skin_net.visualize_motion_weight_vol(os.path.join(logdir, 'vis_motionWeightVol' + str(i).zfill(5) + '.obj')) ######
                torch.save(
                    checkpoint_dict,
                    os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
                )
                tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


if __name__ == "__main__":
    np.random.seed(999)
    torch.random.manual_seed(999)
    main()