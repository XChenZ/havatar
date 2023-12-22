import torch
import numpy as np
from model import nerf_model
from einops import rearrange
from utils.nerf_util import sample_pdf, volume_render_radiance_field
from model.Skinning_Field import Deformation_Field_new
from utils.training_util import get_minibatches
from utils.util import get_box_warp_param, UniformBoxWarp_new


class Trainer(torch.nn.Module):
    def __init__(self, cfg, latent_codes_size=0, freeze_motion=True):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.latent_codes = torch.nn.Parameter(torch.zeros(latent_codes_size, cfg.experiment.latent_code_dim)) if latent_codes_size > 0 else None
        self.model_mode = cfg.experiment.model_mode
        cond_pose = cfg.experiment.cond_pose
        latent_code_dim = (cfg.experiment.latent_code_dim + 12) if cond_pose else cfg.experiment.latent_code_dim
        latent_code_dim += (52 if cfg.experiment.cond_expr else 0)
        self.model_coarse = nerf_model.ConditionalTriplaneNeRFModel_multiRender_split_view(
            XYZ_bounding=cfg.models.coarse.XYZ_bounding,
            triPlane_feat_dim=64,
            rgb_feat_dim=3, triplane_res=128,
            sh_deg=0,
            latent_code_dim=latent_code_dim,
            cond_c_dim=latent_code_dim - cfg.experiment.latent_code_dim)

        self.render_size, self.gen_size = cfg.models.StyleUnet.inp_size, cfg.models.StyleUnet.out_size
        XYZ_bounding_ = cfg.models.coarse.XYZ_bounding
        XYZ_bounding_ = {'X': np.asarray(XYZ_bounding_[0]), 'Y': np.asarray(XYZ_bounding_[1]), 'Z': np.asarray(XYZ_bounding_[2])}
        XYZ_bounding_['Y'][0] = 0.3 * XYZ_bounding_['Y'][1]
        # XYZ_bounding_ = {'X': np.asarray([-1., 1.]), 'Y': np.asarray([0.3, 1.]), 'Z': np.asarray([-0.5, 1.])}
        scales, trans = get_box_warp_param(XYZ_bounding_['X'], XYZ_bounding_['Y'], XYZ_bounding_['Z'])
        self.headpose_skin_net = Deformation_Field_new(gridwarper=UniformBoxWarp_new(scales=scales, trans=trans))
        if freeze_motion:
            self.headpose_skin_net.requires_grad = False

    def nerf_forward(self, **inputs):
        cond_c = inputs['inv_head_T'].view(inputs['inv_head_T'].shape[0], -1)

        self.model_coarse.set_conditional_embedding(front_render_cond=inputs['front_render_cond'], left_render_cond=inputs['left_render_cond'],
                                                    right_render_cond=inputs['right_render_cond'], latents=inputs['latent_code'],
                                                    cond_c=cond_c)

        inv_head_T = inputs['inv_head_T']

        ray_batch, background_prior = inputs['ray_batch'], inputs['background_prior']
        options = self.cfg
        mode = inputs['mode']

        ray_origins, ray_directions = ray_batch[..., :3], ray_batch[..., 3:6]
        viewdirs = ray_directions / ray_directions.norm(p=2, dim=-1).unsqueeze(-1)
        # Cache shapes now, for later restoration.
        restore_shapes = [
            ray_directions.shape[:-1] + (-1,),
            ray_directions.shape[:-1],
            ray_directions.shape[:-1],
            ray_directions.shape[:-1]
        ]
        if getattr(options.nerf, mode).num_fine > 0:
            restore_shapes += restore_shapes[:-1]

        rays = torch.cat((ray_batch, viewdirs), dim=-1)

        split_dim = 1
        batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize // rays.shape[0], dim=split_dim)
        background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize // rays.shape[0], dim=split_dim) if \
            background_prior is not None else background_prior

        pred = [self.predict_and_render_radiance(mode, batch, background_prior[i], inv_head_T=inv_head_T)
                for i, batch in enumerate(batches)]

        synthesized_images = list(zip(*pred))
        synthesized_images = [
            torch.cat(image, dim=split_dim) if image[0] is not None else (None)
            for image in synthesized_images
        ]

        if mode == "validation":
            synthesized_images = [
                image.view(shape) if image is not None else None
                for (image, shape) in zip(synthesized_images, restore_shapes)
            ]
            # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
            # (assuming both the coarse and fine networks are used).
            if getattr(options.nerf, mode).num_fine > 0:
                return tuple(synthesized_images)
            else:
                # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
                # set to None.
                return tuple(synthesized_images + [None, None, None])
        return tuple(synthesized_images)

    def forward(self, **data):
        ray_batch, background_prior = data['ray_batch'], data['background_prior']
        batch_num, rays_num = ray_batch.shape[:2]
        if data['mode'] == 'train':
            latent_code = self.latent_codes[data['fidx']]
        else:
            latent_code = self.latent_codes[0:1]

        latent_code_loss = torch.square(latent_code - self.latent_codes.mean(dim=0, keepdims=True).detach()).mean()

        rgb_coarse, _, acc_coarse, weights, rgb_fine, _, acc_fine = self.nerf_forward(ray_batch=ray_batch, background_prior=background_prior,
                                                                                      latent_code=latent_code, inv_head_T=data['inv_head_T'],
                                                                                      front_render_cond=data['front_render_cond'],
                                                                                      left_render_cond=data['left_render_cond'],
                                                                                      right_render_cond=data['right_render_cond'],
                                                                                      mode=data['mode'])

        if data['render_full_img']:
            render = rgb_fine if rgb_fine is not None else rgb_coarse
            mask = acc_fine if acc_fine is not None else acc_coarse
            render = render.reshape(batch_num, self.render_size, self.render_size, -1).permute(0, 3, 1, 2)  # [B, C, 128, 128]
            mask = mask.reshape(batch_num, self.render_size, self.render_size, -1).permute(0, 3, 1, 2)  # [B, C, 128, 128]
            return render, mask, latent_code_loss
        else:
            return rgb_coarse, _, acc_coarse, weights, rgb_fine, _, acc_fine, latent_code_loss

    def predict_and_render_radiance(self, mode, ray_batch, background_prior, inv_head_T):
        options = self.cfg
        num_batch, num_rays = ray_batch.shape[:2]

        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        bounds = ray_batch[..., 6:8].unsqueeze(-2)
        near, far = bounds[..., 0], bounds[..., 1]

        # when not enabling "ndc".
        t_vals = torch.linspace(0.0, 1.0, getattr(options.nerf, mode).num_coarse, dtype=ro.dtype, device=ro.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals  # [B, N_rays, N_samples]

        if getattr(options.nerf, mode).perturb:
            # Get intervals between samples.
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
            z_vals = lower + (upper - lower) * t_rand

        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]  # [B, N_rays, N_samples, 3]
        viewdirs = ray_batch[..., -3:]

        rot_pts = pts.reshape(num_batch, -1, 3)
        rot_viewdirs = viewdirs.unsqueeze(2).expand(pts.shape).reshape(num_batch, -1, 3)
        rot_pts, rot_viewdirs = self.headpose_skin_net(rot_pts, rot_viewdirs, inv_head_T)
        rot_pts = rot_pts.reshape(pts.shape)

        pts_feat = self.model_coarse.sample_pts_triplane_feat(batch_pts=rearrange(rot_pts, "b r s k -> b (r s) k", k=3))    # [BN, C]
        radiance_field = self.model_coarse(rearrange(rot_pts, "b r s c -> (b r s) c"), pts_feat)
        radiance_field = rearrange(radiance_field, "(b r s) c -> (b r) s c", b=num_batch, r=num_rays)

        background_prior = background_prior.reshape(-1, background_prior.shape[-1])

        z_vals = z_vals.reshape(-1, z_vals.shape[-1])

        (rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse,) = volume_render_radiance_field(
            radiance_field, depth_values=z_vals,
            ray_directions=rd.reshape(-1, rd.shape[-1]),
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            background_prior=background_prior,
            act_feat=False
        )
        rgb_fine, depth_fine, acc_fine = None, None, None
        if getattr(options.nerf, mode).num_fine > 0:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], getattr(options.nerf, mode).num_fine,
                                   det=(getattr(options.nerf, mode).perturb == 0.0))
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat((z_vals[:, ::2], z_samples), dim=-1), dim=-1)
            # z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
            z_vals = z_vals.reshape(num_batch, num_rays, z_vals.shape[-1])
            pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

            viewdirs = ray_batch[..., -3:]

            rot_pts = pts.reshape(num_batch, -1, 3)
            rot_viewdirs = viewdirs.unsqueeze(2).expand(pts.shape).reshape(num_batch, -1, 3)
            rot_pts, rot_viewdirs = self.headpose_skin_net(rot_pts, rot_viewdirs, inv_head_T)
            rot_pts = rot_pts.reshape(pts.shape)

            pts_feat = self.model_coarse.sample_pts_triplane_feat(batch_pts=rearrange(rot_pts, "b r s k -> b (r s) k", k=3))    # [BN, C]
            radiance_field = self.model_coarse(rearrange(rot_pts, "b r s c -> (b r s) c"), pts_feat)
            radiance_field = rearrange(radiance_field, "(b r s) c -> (b r) s c", b=num_batch, r=num_rays)
            z_vals = z_vals.reshape(-1, z_vals.shape[-1])
            rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field(  # added use of weights
                radiance_field,
                depth_values=z_vals,
                ray_directions=rd.reshape(-1, rd.shape[-1]),
                radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
                background_prior=background_prior,
                act_feat=False
            )
            return rgb_coarse.reshape(num_batch, num_rays, -1), depth_coarse.reshape(num_batch, num_rays, -1),\
                   acc_coarse.reshape(num_batch, num_rays, -1), weights.max(dim=-1)[0].reshape(num_batch, num_rays, -1), \
                   rgb_fine.reshape(num_batch, num_rays, -1), depth_fine.reshape(num_batch, num_rays, -1), \
                   acc_fine.reshape(num_batch, num_rays, -1)  # changed last return val to fine_weights
        else:
            return rgb_coarse.reshape(num_batch, num_rays, -1), depth_coarse.reshape(num_batch, num_rays, -1), \
                acc_coarse.reshape(num_batch, num_rays, -1), weights.max(dim=-1)[0].reshape(num_batch, num_rays, -1), \
                rgb_fine, depth_fine, acc_fine


