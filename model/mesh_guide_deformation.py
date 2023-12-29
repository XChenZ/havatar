import torch
import torch.nn as nn
from model.network.embedder import get_embedder

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
