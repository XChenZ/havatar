import torch
import torch.nn as nn
from model.network.embedder import get_embedder
from utils.pytorch3d_util import point2point_distance
from flame.FLAME import FLAME

class Morphable_model(object):
    def __init__(self, model_type='Flame', device='cuda:0'):
        assert model_type in ['Flame']
        self.model_type = model_type
        if self.model_type == 'Flame':
            self.model = FLAME().to(device)
            self.n_shape, self.n_expr, self.n_pose = 100, 50, 15
            self.cano_verts = self.model.v_template.unsqueeze(0) * 4

    def get_verts(self, param_dict):
        if self.model_type == 'Flame':
            verts_p = self.model(shape_params=param_dict['shape'], expression_params=param_dict['exp'],
                                 full_pose=param_dict['full_pose'], only_vert=True)
            # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly # IMAvatar
            verts_p *= 4
            return verts_p

    def split_params(self, param):
        if self.model_type == 'Flame':
            assert param.shape[-1] == self.n_shape + self.n_expr + self.n_pose
            param_dict = {
                'shape': param[..., :self.n_shape],
                'exp': param[..., self.n_shape:self.n_shape+self.n_expr],
                'full_pose':  param[..., self.n_shape+self.n_expr:self.n_shape+self.n_expr+self.n_pose]
            }
            return param_dict

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 0.)
        m.bias.data.fill_(0.)

class NonRigid_Deformation_Net(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=10, latent_code_dim=32, hidden_dim=128, layers_num=8):
        super(NonRigid_Deformation_Net, self).__init__()
        self.name = 'NonRigid_Deformation_Net'
        self.pos_embedder, self.dim_xyz = get_embedder(multires=num_encoding_fn_xyz, input_dims=3, include_input=True)
        self.pos_embedder2, self.dim_xyz2 = get_embedder(multires=2, input_dims=3, include_input=True)
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        lin = torch.nn.Linear(self.dim_xyz + self.dim_xyz2 + self.dim_latent_code, hidden_dim)
        self.layers_xyz.append(lin)
        for i in range(1, layers_num):
            if i == layers_num//2:
                lin = torch.nn.Linear(self.dim_xyz + self.dim_xyz2 + self.dim_latent_code + hidden_dim, hidden_dim)
            else:
                lin = torch.nn.Linear(hidden_dim, hidden_dim)
            self.layers_xyz.append(lin)
        self.fc_out = torch.nn.Linear(hidden_dim, 3)
        self.relu = torch.nn.functional.relu
        self.apply(init_weights)

    def forward(self, inp, latent_code, alpha=None):
        xyz, xyz_3dmm = inp[..., :3], inp[..., 3:]
        xyz_emb = self.pos_embedder(xyz, alpha)
        xyz_3dmm_emb = self.pos_embedder2(xyz_3dmm)
        initial = torch.cat((xyz_emb, xyz_3dmm_emb, latent_code), dim=-1)
        x = initial
        out_feat_ls = []
        for i in range(len(self.layers_xyz)):
            if i == len(self.layers_xyz)//2:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            out_feat_ls.append(x)
            x = self.relu(x)
        out = self.fc_out(x)
        return out, out_feat_ls


class MeshGuide_Deformation_Field(torch.nn.Module):
    def __init__(self, model_type):
        super(MeshGuide_Deformation_Field, self).__init__()
        self.MModel = Morphable_model(model_type)
        # self.register_buffer('identity_trans', torch.eye(4, dtype=torch.float32)[:, :-1])
        self.deformation_mlp = NonRigid_Deformation_Net()
        self.pe_alpha = None

    def forward(self, pts, MModel_params, motion_latent):
        '''
        :param pts: [B, P, 3]
        :MModel_params: [B, P, C]
        :return:
        '''
        param_dict = MModel_params if type(MModel_params) is dict else self.MModel.split_params(MModel_params)
        deformed_mesh_pts = self.MModel.get_verts(param_dict)   # [B, N, 3]
        mesh_pts_shift = self.MModel.cano_verts - deformed_mesh_pts # [B, N, 3]

        point_to_mesh_dists, idxs, _ = point2point_distance(pts, deformed_mesh_pts)

        weight = 1. / torch.exp(point_to_mesh_dists).unsqueeze(-1) # [B, P, 1]
        pts_mesh_guide_shifts = []
        for b in range(pts.shape[0]):
            pts_mesh_guide_shift = mesh_pts_shift[b][idxs[b]]  # [P, 3]
            pts_mesh_guide_shifts.append(pts_mesh_guide_shift)
        pts_mesh_guide_shifts = torch.stack(pts_mesh_guide_shifts, dim=0)   # [B, P, 3]
        pts_mesh_guide_shifts = pts_mesh_guide_shifts * weight

        if False:
            print(pts.shape, deformed_mesh_pts.shape)
            from utils.util import save_obj
            save_obj('test.obj', v=deformed_mesh_pts[0].cpu().numpy())
            save_obj('test0.obj', v=self.MModel.cano_verts[0].cpu().numpy())

            pts = deformed_mesh_pts
            point_to_mesh_dists, idxs, _ = point2point_distance(pts, deformed_mesh_pts)

            weight = 1. / torch.exp(point_to_mesh_dists).unsqueeze(-1)  # [B, P, 1]
            pts_mesh_guide_shifts = []
            for b in range(pts.shape[0]):
                pts_mesh_guide_shift = mesh_pts_shift[b][idxs[b]]  # [P, 3]
                pts_mesh_guide_shifts.append(pts_mesh_guide_shift)
            pts_mesh_guide_shifts = torch.stack(pts_mesh_guide_shifts, dim=0)  # [B, P, 3]
            pts_mesh_guide_shifts = pts_mesh_guide_shifts * weight
            save_obj('test1.obj', v=(pts + pts_mesh_guide_shifts)[0].cpu().numpy())
            exit(0)

        pts_mlp_shifts, pts_motion_feature_ls = self.deformation_mlp(torch.cat([pts, pts_mesh_guide_shifts], dim=-1),
                                                                     motion_latent, self.pe_alpha)
        pts_shifts = pts_mesh_guide_shifts + pts_mlp_shifts

        return pts + pts_shifts, pts_motion_feature_ls[-1]
