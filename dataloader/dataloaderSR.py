"""Data loader"""

from __future__ import division, print_function

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import json
import copy
from dataloader import data_util, dist_util

torch.utils.backcompat.broadcast_warning.enabled = True


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


class MultiView_ImgDataset(Dataset):
    def __init__(self, split_file, mode, options, down_sample=1.0, white_bg=True):
        super(MultiView_ImgDataset, self).__init__()
        self.mode = mode
        assert mode in ['train', 'val', 'test']
        self.options = options
        self.down_sample = down_sample
        self.white_bg = white_bg
        assert os.path.exists(split_file), split_file
        meta = json.loads(open(split_file).read())
        self.img_w, self.img_h = meta['img_res'], meta['img_res']

        self.mv_intrinsics = np.asarray(meta["mutiview_intr_ls"], dtype=np.float32)
        if self.down_sample < 1:
            self.mv_intrinsics[:, :2] = self.mv_intrinsics[:, :2] * self.down_sample
            self.img_w, self.img_h = int(self.img_w * self.down_sample), int(self.img_h * self.down_sample)

        self.view_num = self.mv_intrinsics.shape[0]
        self.load_background(bg_paths=meta["bg_path"] if 'bg_path' in meta.keys() else None, white_bg=white_bg)

        self.frames = []
        for tmp_frame_dict in meta['frames']:
            for vidx in range(len(tmp_frame_dict["mutiview_info_ls"])):
                if tmp_frame_dict["mutiview_info_ls"][vidx]['view_name'] == '8': continue  #####
                tmp_frame_dict_ = copy.deepcopy(tmp_frame_dict)
                tmp_frame_dict_['vidx'] = vidx
                self.frames.append(tmp_frame_dict_)

        self.frames.sort(key=lambda x:x['fidx'])
        self.coords_yx = torch.stack(data_util.meshgrid_xy(torch.arange(self.img_h), torch.arange(self.img_w)), dim=-1).reshape((-1, 2)).flip([-1])  # YX

    def load_background(self, bg_paths, white_bg):
        self.bgs = []
        if white_bg:
            for _ in range(self.view_num):
                background = torch.ones((self.img_h, self.img_w, 3), dtype=torch.float32)
                self.bgs.append(background)
        else:
            for bg_path in bg_paths:
                background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
                if self.down_sample < 1:
                    background = cv2.resize(background, dsize=(self.img_h, self.img_w), interpolation=cv2.INTER_AREA)
                background = torch.from_numpy(np.asarray(background).astype(np.float32)) / 255
                self.bgs.append(background)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_dict = self.frames[idx]
        data_dict = self.load_data(frame_dict)
        # data_dict['idx'] = idx
        return idx, data_dict

    def load_data(self, frame_dict):
        frame_ind = frame_dict['fidx']
        mv_rays_gt_color, mv_rays, mv_depth = [], [], []
        coords = self.coords_yx
        vidx_ls = []
        if True:
            view_idx = frame_dict['vidx']
            view_dict = frame_dict["mutiview_info_ls"][frame_dict['vidx']]
            vidx_ls.append(int(view_dict['view_name']))
            pose = torch.from_numpy(np.asarray(view_dict["transform_matrix"], dtype=np.float32))
            if 'cam_K' in view_dict.keys():
                cam_K = np.asarray(view_dict['cam_K'], dtype=np.float32)# if 'cam_K' in view_dict.keys() else
                if self.down_sample < 1:
                    cam_K[:2] = cam_K[:2] * self.down_sample
            else:
                cam_K = self.mv_intrinsics[view_idx]

            ray_origins, ray_directions = data_util.get_rays(self.img_h, self.img_w, cam_K, pose[:3, :4], normalize=True)
            dx = torch.sqrt(torch.sum((ray_directions[:, :-1, :] - ray_directions[:, 1:, :]) ** 2, -1))  # [H, W-1]
            dx = torch.cat([dx, dx[:, -2:-1]], 1)  # [H, W]

            select_inds = coords
            if self.mode == 'train':
                mask = cv2.cvtColor(cv2.imread(view_dict["mask_path"]), cv2.COLOR_BGR2RGB)
                mask = (mask[:, :, 0] > 127).astype(np.float32)
                if self.down_sample < 1:
                    mask_ds = cv2.resize(mask, dsize=(0, 0), fx=self.down_sample, fy=self.down_sample, interpolation=cv2.INTER_AREA)
                else:
                    mask_ds = mask
                assert (self.img_h, self.img_w) == mask_ds.shape[:2]
                ray_m = torch.from_numpy(mask_ds)[select_inds[:, 0], select_inds[:, 1]].unsqueeze(-1) if mask_ds is not None else None
            else:
                ray_m = None
            ray_o = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_d = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            ray_bg = self.bgs[view_idx][select_inds[:, 0], select_inds[:, 1], :]


            dist = torch.norm(torch.from_numpy(np.asarray(view_dict["transform_matrix_ori"], dtype=np.float32))[:3, -1].expand(ray_d.shape),
                              dim=-1, keepdim=True)
            ray_near = (dist + self.options.dataset.near * self.options.dataset.length) * torch.ones_like(ray_d[..., :1])
            ray_far = (dist + self.options.dataset.far * self.options.dataset.length) * torch.ones_like(ray_d[..., :1])

            mv_ray = torch.cat([ray_o, ray_d, ray_near, ray_far, ray_bg, ray_m], dim=1) if ray_m is not None \
                else torch.cat([ray_o, ray_d, ray_near, ray_far, ray_bg], dim=1)
            mv_rays.append(mv_ray)

            if not self.mode == 'test': # 非test情况下load gt img(原尺寸)
                img = cv2.cvtColor(cv2.imread(view_dict["file_path"]), cv2.COLOR_BGR2RGB)
                if self.white_bg: img[mask==0] = 255
                img_tensor = torch.from_numpy((np.array(img) / 255.0).astype(np.float32))  # * 2. -1.  # [-1, 1]
                coords = torch.stack(data_util.meshgrid_xy(torch.arange(img_tensor.shape[0]), torch.arange(img_tensor.shape[1])), dim=-1).reshape((-1, 2)).flip([-1])
                mv_rays_gt_color.append(img_tensor[coords[:, 0], coords[:, 1], :])

        if self.mode == 'test':
            data_dict = {
                'fidx': frame_ind,
                'vidx': vidx_ls,
                'mv_rays': torch.cat(mv_rays, 0),
            }
        else:
            data_dict = {
                'mv_rays_gt_color': torch.cat(mv_rays_gt_color, 0),
                'mv_rays': torch.cat(mv_rays, 0),
            }

        head_transformation = np.asarray(frame_dict['head_transformation']).astype(np.float32)[:3]  # 右乘

        data_dict['front_render_cond'] = make_render_cond_(normal_path=os.path.join(frame_dict['inst_dir'], 'ortho_front_normal_256_baseGama.png'),
                                                           render_path=os.path.join(frame_dict['inst_dir'], 'ortho_front_render_256_baseGama.png'),
                                                           res=self.options.dataset.cond_render_res)
        data_dict['left_render_cond'] = make_render_cond_(normal_path=os.path.join(frame_dict['inst_dir'], 'ortho_left_normal_256_baseGama.png'),
                                                          render_path=os.path.join(frame_dict['inst_dir'], 'ortho_left_render_256_baseGama.png'),
                                                          res=self.options.dataset.cond_render_res)
        data_dict['right_render_cond'] = make_render_cond_(normal_path=os.path.join(frame_dict['inst_dir'], 'ortho_right_normal_256_baseGama.png'),
                                                           render_path=os.path.join(frame_dict['inst_dir'], 'ortho_right_render_256_baseGama.png'),
                                                           res=self.options.dataset.cond_render_res)
        rotation, translation = head_transformation.T[:3, :3], head_transformation.T[-1:]
        data_dict['inv_head_T'] = torch.from_numpy(np.concatenate([np.linalg.inv(rotation), -translation], 0))  # [4, 3]

        return data_dict

def make_render_cond_(normal_path, render_path, res):
    normal = cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB)
    if not res == normal.shape[0]:
        normal = cv2.resize(normal, dsize=(res, res), interpolation=cv2.INTER_LINEAR)
    mask = (np.linalg.norm(normal, axis=-1) > 0.).astype(np.float32)
    render = cv2.cvtColor(cv2.imread(render_path), cv2.COLOR_BGR2RGB)
    if not res == render.shape[0]:
        render = cv2.resize(render, dsize=(res, res), interpolation=cv2.INTER_LINEAR)
    render_cond = torch.from_numpy(
        np.concatenate([render.astype(np.float32) / 255.0, normal.astype(np.float32) / 255.0, mask[:, :, None]], axis=-1))  # [H, W, 7]
    return render_cond

class Loader(DataLoader):
    def __init__(self, split_file, options, mode='train', batch_size=4, num_workers=0, down_sample=1.0, distributed=False, white_bg=True, shuffle=None):
        self.dataset = MultiView_ImgDataset(split_file, mode, options, down_sample, white_bg=white_bg)
        self.batch_size = batch_size
        if shuffle is None: shuffle = mode=='train'
        self.sampler = dist_util.data_sampler(self.dataset, shuffle=shuffle, distributed=distributed)
        super(Loader, self).__init__(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=self.sampler,
                                     num_workers=num_workers,
                                     worker_init_fn=worker_init_fn,
                                     pin_memory=True,
                                     drop_last=True)