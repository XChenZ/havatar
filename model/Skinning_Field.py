import torch
import torch.nn as nn
from model.network.embedder import get_embedder
from utils.util import voxel_feature
from utils.util import save_obj_data, make_volume_pts
from model.network.voxel_encoder import VolumeDecoder
from utils.util import UniformBoxWarp
from tqdm import tqdm
import copy

rot_Types = {
    'mat': {'dim': 9},
    'euler': {'dim': 3},
    'quat': {'dim': 4},

}

class NonRigid_Deformation_Net(torch.nn.Module):
    def __init__(self):
        super(NonRigid_Deformation_Net, self).__init__()
        self.pos_embedder, self.dim_xyz = get_embedder(multires=8, input_dims=3, include_input=True)
        self.layers_xyz = torch.nn.ModuleList([nn.Linear(self.dim_xyz + self.dim_latent_code, 128)] +
                                              [nn.Linear(128, 128), nn.Linear(128, 128)])
        self.relu = torch.nn.functional.relu
        self.to_xyz = nn.Linear(128, 3)

    def forward(self, pts, pose):
        '''
        :param pts: [B, N, 3]
        :param pose_rot: [B, 4, 4]
        :return:
        '''
        pts_emb = self.pos_embedder(pts)
        print(pose[0, -1], pose[0, :, -1])
        exit(0)
        x = torch.cat([pts_emb, pose.view(-1, 1, 12).expand(-1, pts_emb.shape[1], -1)], dim=-1)  # [B, N, C]
        for i, l in enumerate(self.layers_xyz):
            x = self.layers_xyz[i](x)
            x = self.relu(x)
        return pts + self.to_xyz(x)


class Deformation_Field_new(torch.nn.Module):
    def __init__(self, gridwarper=None, options=None, need_nr=False):
        super(Deformation_Field_new, self).__init__()
        self.canonical_Wvolume = VolumeDecoder(num_in=options['init_length'] if options is not None else 1024, num_out=1,
                                               final_res=options['vol_res'] if options is not None else 64,#16,
                                               up_mode=options['up_mode'] if options is not None else 'upsample')  # canonical_skinningWeight_volume
        self.gridwarper = UniformBoxWarp(scales=(1 / 2.5, 1 / 2.5, 1 / 2.0), trans=(0., -0., -0.2)) if gridwarper is None else copy.deepcopy(gridwarper)
        self.register_buffer('identity_trans', torch.eye(4, dtype=torch.float32)[:, :-1])
        # self.update_head_T = None
        self.nr_motion_field = None
        if need_nr:
            self.nr_motion_field = NonRigid_Deformation_Net()

    def sample_volume(self, pts, padding_mode='border'):
        # pts [N, 3]
        vol = self.canonical_Wvolume()  # [1, 2, 32, 32, 32]
        return voxel_feature(xyz=self.gridwarper(pts.unsqueeze(0)), volume_feat=vol[:, 0:1], padding_mode=padding_mode)[0]     # [N, 1]

    def forward(self, pts, pts_view, inv_Trans):
        '''
        :param pts: [B, N, 3]
        :param pose_rot: [B, 3, 3]
        :return:
        '''
        batch_size = inv_Trans.shape[0]
        inv_Trans_ls = [self.identity_trans.unsqueeze(0).expand(batch_size, -1, -1), inv_Trans]
        pts_inv_ls = []
        w_c = self.canonical_Wvolume().expand(batch_size, -1, -1, -1, -1)  # [1, 2, 32, 32, 32] ### 分chunk时会重复计算
        pts_wc = []
        for i in range(len(inv_Trans_ls)):
            inv_T = inv_Trans_ls[i]
            pts_inv = torch.matmul(pts + inv_T[:, -1:], inv_T[:, :3, :3])
            pts_inv_ls.append(pts_inv)
            pts_wc.append(voxel_feature(xyz=self.gridwarper(pts_inv), volume_feat=w_c[:, i:i + 1]))  # [B, N, 1]
        pts_wc = torch.cat(pts_wc, -1)  # [B, N, 2]
        pts_w = pts_wc / (pts_wc.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N, 2]
        pts_c, pts_view_c = [], []
        for i in range(len(pts_inv_ls)):
            pts_c.append(pts_w[:, :, i:i + 1] * pts_inv_ls[i])
            if pts_view is not None:
                pts_view_c.append(pts_w[:, :, i:i + 1] * (torch.matmul(pts_view, inv_Trans_ls[i][:, :3, :3])))
            else:
                pts_view_c.append(0)
        out_pts_c, out_pts_view_c = sum(pts_c), sum(pts_view_c)
        if self.nr_motion_field is not None:
            out_pts_c = self.nr_motion_field(out_pts_c, inv_Trans)
        return out_pts_c, out_pts_view_c#, pts_wc[0]

    # def forward(self, pts, pts_view, inv_Trans):
    #     '''
    #     :param pts: [B, N, 3]
    #     :param pose_rot: [B, 3, 3]
    #     :return:
    #     '''
    #     batch_size = inv_Trans.shape[0]
    #     inv_Trans_ls = [self.identity_trans.unsqueeze(0).expand(batch_size, -1, -1), inv_Trans]
    #     pts_inv_ls = []
    #     self.w_c = self.canonical_Wvolume().expand(batch_size, -1, -1, -1, -1)  # [1, 2, 32, 32, 32]
    #     pts_wc = []
    #     for i in range(len(inv_Trans_ls)):
    #         pts_inv = torch.matmul(pts + inv_Trans_ls[i][:, -1:], inv_Trans_ls[i][:, :3, :3])
    #         pts_inv_ls.append(pts_inv)
    #         pts_wc.append(voxel_feature(xyz=self.gridwarper(pts_inv), volume_feat=self.w_c[:, i:i+1])) # [B, N, 1]
    #     self.pts_wc = torch.cat(pts_wc, -1)  # [B, N, 2]
    #     self.pts_w = self.pts_wc / (self.pts_wc.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N, 2]
    #     if self.pts_wc.requires_grad:
    #         self.pts_wc.retain_grad()   ######
    #         self.w_c.retain_grad()
    #         self.pts_w.retain_grad()
    #     pts_c, pts_view_c = [], []
    #     for i in range(len(pts_inv_ls)):
    #         pts_c.append(self.pts_w[:, :, i:i+1] * pts_inv_ls[i])
    #         pts_view_c.append(self.pts_w[:, :, i:i+1] * (torch.matmul(pts_view, inv_Trans_ls[i][:, :3, :3])))
    #     return sum(pts_c), sum(pts_view_c)

    def pretrain_wc(self, num_iter=1, lr=1e-3, save_path=None, pose_space=False, vol_thr=None):
        if vol_thr is None: vol_thr = [[-0.5, 0.5], [-0.8, 0.5], [-0.3, 1.0]]
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=lr)
        bar = tqdm(range(num_iter))
        device = self.identity_trans.device
        for i in bar:
            pts_vol = make_volume_pts(steps=20, perturb=True, gridwarper=self.gridwarper).to(device)
            pts_wc_gt = torch.zeros(pts_vol.shape[0], 1).to(device)
            pts_wc_gt[(pts_vol[:, 0] > vol_thr[0][0]) * (pts_vol[:, 0] < vol_thr[0][1]) *
                      (pts_vol[:, 1] > vol_thr[1][0]) * (pts_vol[:, 1] < vol_thr[1][1]) *
                      (pts_vol[:, 2] > vol_thr[2][0]) * (pts_vol[:, 2] < vol_thr[2][1])] = 1
            wc_predict = self.canonical_Wvolume()
            pts_wc = voxel_feature(xyz=self.gridwarper(pts_vol.unsqueeze(0)), volume_feat=wc_predict[:, 0:1] if pose_space else wc_predict[:, 1:])  # [B, N, 1]
            # print(pts_wc[0].max().item(), pts_wc[0].min().item(), pts_wc_gt.max().item(), pts_wc_gt.min().item())
            pts_wc = torch.clamp(pts_wc, 0., 1.)
            loss = torch.nn.functional.binary_cross_entropy(pts_wc[0], pts_wc_gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 ==0:
                bar.set_description(str(loss.item()))

        if save_path is not None:
            torch.save(self.canonical_Wvolume(), save_path)
            # torch.save(self.canonical_Wvolume.state_dict(), save_path)

    def visualize_motion_weight_vol(self, path):
        pts_vol = make_volume_pts(steps=20, perturb=False, gridwarper=self.gridwarper).to('cuda')
        wc_predict = self.canonical_Wvolume()
        pts_wc = voxel_feature(xyz=self.gridwarper(pts_vol.unsqueeze(0)), volume_feat=wc_predict[:, 1:])  # [B, N, 1]
        pts_c = pts_wc[..., 0:1].detach().cpu() * torch.ones(1, 1, 3)
        save_obj_data({'v': pts_vol.reshape(-1, 3).detach().cpu().numpy(), 'vc': pts_c.reshape(-1, 3).numpy()}, path)
