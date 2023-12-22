import torch
import torch.nn as nn
from utils.sh_util import eval_sh
from model.network.embedder import get_embedder
import numpy as np
from model.styleUnet import StyleGAN_zxc, StyleGAN_zxc_twoHead
from utils.util import create_UniformBoxWarp, sample_from_triplane_new


class ConditionalTriplaneNeRFModel_multiRender_split_view(torch.nn.Module):
    def __init__(self, XYZ_bounding, num_encoding_fn_xyz=8, latent_code_dim=32, triPlane_feat_dim=32, rgb_feat_dim=32, triplane_res=256, use_emb=True,
                 enc_mode='split', sh_deg=2, cond_latent=True, cond_c_dim=0):
        super(ConditionalTriplaneNeRFModel_multiRender_split_view, self).__init__()
        self.name = 'ConditionalTriplaneNeRFModel_multiRender_split_view'

        self.pos_embedder, self.dim_xyz = get_embedder(multires=num_encoding_fn_xyz, input_dims=3, include_input=False)
        # self.dir_embedder, self.dim_dir = get_embedder(multires=num_encoding_fn_dir, input_dims=3, include_input=include_input_dir)
        self.sh_deg = sh_deg
        self.use_sh = self.sh_deg >= 1

        include_xyz = self.dim_xyz if use_emb else 0
        self.dim_latent_code = 0 if cond_latent else latent_code_dim
        self.triPlane_feat_dim = triPlane_feat_dim
        self.rgb_feat_dim = rgb_feat_dim * (self.sh_deg + 1) ** 2
        self.use_emb = use_emb
        self.cond_latent = cond_latent
        assert enc_mode in ['split', 'shared_backbone', 'two_head']
        self.shared_backbone = enc_mode == 'shared_backbone'
        self.two_head = enc_mode == 'two_head'
        self.cond_c_dim = cond_c_dim

        if self.shared_backbone:
            self.XY_gen = StyleGAN_zxc(out_ch=triPlane_feat_dim * 2, out_size=triplane_res, style_dim=latent_code_dim, middle_size=16,
                                       n_mlp=4, inp_size=256, inp_ch=7+13)
        elif self.two_head:
            self.XY_gen = StyleGAN_zxc_twoHead(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=8,
                                               split_size=32, zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=[7, 13])
        else:
            self.XY_gen = StyleGAN_zxc(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=16,
                                       zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=7)
            self.YZ_gen = StyleGAN_zxc(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=16,
                                       zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=13)

        self.gridwarper = create_UniformBoxWarp(XYZ_bounding)

        self.layers_xyz = torch.nn.ModuleList([nn.Linear(2 * self.triPlane_feat_dim + self.dim_latent_code + include_xyz, 128)] +
                                              [nn.Linear(128, 128)])
        self.fc_alpha = torch.nn.Linear(128, 1)
        # self.fc_rgb = torch.nn.Linear(128, self.rgb_feat_dim)
        self.fc_rgbFeat = torch.nn.Linear(128, 64)  ####
        self.fc_rgb = torch.nn.Linear(64, self.rgb_feat_dim)   ####

        self.relu = torch.nn.functional.relu
        self.pts_triPlane_feat, self.pts_mask = None, None
        if not self.cond_latent:
            self.register_buffer('zero_latent', torch.zeros(latent_code_dim, dtype=torch.float32).reshape(1, -1))

    def set_conditional_embedding(self, **canonical_condition):
        if 'latents' in canonical_condition.keys():
            latents = canonical_condition['latents']    # [B, L]
            cond_c = canonical_condition['cond_c'].reshape(latents.shape[0], -1)
            if self.cond_latent:
                inp_latents = [torch.cat([latents, cond_c], -1)] if self.cond_c_dim > 0 else [latents]
            else:
                inp_latents = [self.zero_latent.expand(latents.shape[0], -1)]
        else:
            inp_latents = None

        front_render_cond = canonical_condition['front_render_cond']    # [B, 7, H, W]
        left_render_cond = canonical_condition['left_render_cond'].flip(dims=[3])  # [B, 7, H, W]   #右平面满足左上角(-1,-1)，右下角(1, 1)
        right_render_cond = canonical_condition['right_render_cond']  # [B, 7, H, W]
        if left_render_cond.shape[1] > 3: left_render_cond = left_render_cond[:, :-1]  # 如果left_render_cond只含三通道，就没有mask了
        if self.shared_backbone:
            conditonplane_embedding = self.XY_gen(inp_latents, torch.cat([front_render_cond, left_render_cond, right_render_cond], dim=1))
            if type(conditonplane_embedding) is not torch.Tensor: conditonplane_embedding = conditonplane_embedding[0]
            conditonXYplane_embedding, conditonYZplane_embedding = conditonplane_embedding[:, :self.triPlane_feat_dim], \
                                                                   conditonplane_embedding[:, self.triPlane_feat_dim:]
        elif self.two_head:
            conditonXYplane_embedding, conditonYZplane_embedding = self.XY_gen(
                inp_latents, [front_render_cond, torch.cat([left_render_cond, right_render_cond], dim=1)])
        else:
            conditonXYplane_embedding, _ = self.XY_gen(inp_latents, front_render_cond)
            conditonYZplane_embedding, _ = self.YZ_gen(inp_latents, torch.cat([left_render_cond, right_render_cond], dim=1))

        conditionPlanes_feat = torch.stack([conditonXYplane_embedding, conditonYZplane_embedding], dim=0)   # [2, B, C, H, W]]
        self.triPlane_embeddings = conditionPlanes_feat

    def sample_pts_triplane_feat(self, batch_pts, bidx=None):
        '''
        :param batch_pts: [B, N, 3]
        :param return_feat:
        '''
        inp_pts = self.gridwarper(batch_pts)
        if bidx == None:
            pts_triPlane_feat = sample_from_triplane_new(inp_pts, self.triPlane_embeddings, padding_mode='zeros')    # [B, N, C, 3]
        else:
            assert len(bidx) == batch_pts.shape[0]
            pts_triPlane_feat = sample_from_triplane_new(inp_pts, self.triPlane_embeddings[:, bidx], padding_mode='zeros')  # [N, C, 3]
        return pts_triPlane_feat.reshape(-1, pts_triPlane_feat.shape[-1] * pts_triPlane_feat.shape[-2]) # [BN, C * 3]

    def forward(self, inp, pts_feat):
        xyz, dirs = inp[..., :3], inp[..., 3:]
        xyz_emb = self.pos_embedder(xyz)
        pts_feat = torch.cat([pts_feat, xyz_emb], -1)
        x = pts_feat
        for i, l in enumerate(self.layers_xyz):
            x = self.layers_xyz[i](x)
            x = self.relu(x)
        alpha = self.fc_alpha(x)
        x = self.fc_rgbFeat(x)
        sh = self.fc_rgb(x)  # [N, C* (deg+1)**2]
        if self.sh_deg == 0:
            rgb = sh
        else:
            rgb = eval_sh(self.sh_deg, sh.reshape(sh.shape[0], -1, (self.sh_deg + 1) ** 2), dirs)
        rgb = torch.cat([rgb, x], dim=-1)
        return torch.cat((rgb, alpha), dim=-1)
