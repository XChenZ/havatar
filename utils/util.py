import numpy as np
import torch

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
def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')
        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def save_obj_data(model, filename, log=True):
    import numpy as np
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                assert model['vc'].size == model['v'].size
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[2], vc[1], vc[0]))
            else:
                for v in model['v']:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if log:
        print("Saved mesh as " + filename)


def save_obj(path, v, f=None, c=None):
    with open(path, 'w') as file:
        for i in range(len(v)):
            if c is not None:
                file.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
            else:
                file.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], 1, 1, 1))

        file.write('\n')
        if f is not None:
            for i in range(len(f)):
                file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()


def get_box_warp_param(X_bounding, Y_bounding, Z_bounding):
    fx = 2 / (X_bounding[1] - X_bounding[0])
    cx = fx * (X_bounding[0] + X_bounding[1]) * 0.5
    fy = 2 / (Y_bounding[1] - Y_bounding[0])
    cy = fy * (Y_bounding[0] + Y_bounding[1]) * 0.5
    fz = 2 / (Z_bounding[1] - Z_bounding[0])
    cz = fz * (Z_bounding[0] + Z_bounding[1]) * 0.5
    return (float(fx), float(fy), float(fz)), (float(-cx), float(-cy), float(-cz))


def create_UniformBoxWarp(XYZ_bounding):
    XYZ_bounding_ = {'X': np.asarray(XYZ_bounding[0]), 'Y': np.asarray(XYZ_bounding[1]), 'Z': np.asarray(XYZ_bounding[2])}
    scales, trans = get_box_warp_param(XYZ_bounding_['X'], XYZ_bounding_['Y'], XYZ_bounding_['Z'])
    return UniformBoxWarp_new(scales=scales, trans=trans)


class UniformBoxWarp(torch.nn.Module):
    def __init__(self, scales, trans):
        super().__init__()
        self.register_buffer('scale_factor', torch.tensor(scales, dtype=torch.float32).reshape(1, 3))
        self.register_buffer('trans_factor', torch.tensor(trans, dtype=torch.float32).reshape(1, 3))
        # self.scale_factor = torch.tensor(scales, dtype=torch.float32).reshape(1, 3)
        # self.trans_factor = torch.tensor(trans, dtype=torch.float32).reshape(1, 3)

    def inv_trans(self, pts):
        pts_ = (pts * 0.5 - self.trans_factor.to(pts.device)) / self.scale_factor.to(pts.device)
        return pts_

    def forward(self, coordinates):
        # [N, 3] OR [B, N, 3]
        scale_factor = self.scale_factor.unsqueeze(0) if coordinates.ndim==3 else self.scale_factor
        trans_factor = self.trans_factor.unsqueeze(0) if coordinates.ndim==3 else self.trans_factor
        return 2.0 * ((coordinates * scale_factor) + trans_factor)


class UniformBoxWarp_new(torch.nn.Module):
    def __init__(self, scales, trans):
        super().__init__()
        self.register_buffer('scale_factor', torch.tensor(scales, dtype=torch.float32).reshape(1, 3))
        self.register_buffer('trans_factor', torch.tensor(trans, dtype=torch.float32).reshape(1, 3))
        # self.scale_factor = torch.tensor(scales, dtype=torch.float32).reshape(1, 3)
        # self.trans_factor = torch.tensor(trans, dtype=torch.float32).reshape(1, 3)

    def inv_trans(self, coordinates):
        # [N, 3] OR [B, N, 3]
        scale_factor = self.scale_factor.unsqueeze(0) if coordinates.ndim==3 else self.scale_factor
        trans_factor = self.trans_factor.unsqueeze(0) if coordinates.ndim==3 else self.trans_factor
        if type(coordinates) is np.ndarray:
            out_pts = (coordinates - trans_factor.cpu().numpy()) * (1. / scale_factor.cpu().numpy())
        else:
            out_pts = (coordinates - trans_factor.to(coordinates.device)) * (1. / scale_factor.to(coordinates.device))
        return out_pts

    def forward(self, coordinates):
        # [N, 3] OR [B, N, 3]
        scale_factor = self.scale_factor.unsqueeze(0) if coordinates.ndim==3 else self.scale_factor
        trans_factor = self.trans_factor.unsqueeze(0) if coordinates.ndim==3 else self.trans_factor
        return (coordinates * scale_factor) + trans_factor


def make_volume_pts(steps=50, perturb=False, gridwarper=None, z_scale=1.0):
    # from nerf.models import UniformBoxWarp
    x_coords = torch.linspace(-1.0, 1.0, steps=steps, dtype=torch.float32)
    y_coords = x_coords.clone()
    z_coords = torch.linspace(-1.0, 1.0, steps=int(steps * z_scale), dtype=torch.float32)
    xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # (volume_res, volume_res, volume_res)
    pts_vol = torch.stack([xv, yv, zv], dim=-1)  # (volume_res, volume_res, volume_res, 3)
    pts = pts_vol.reshape(-1, 3)

    if perturb:
        pts = pts + torch.rand_like(pts) * (2 / (steps - 1))

    if gridwarper is not None:
        pts = gridwarper.inv_trans(pts)
    # print(pts[:, 0].max(), pts[:, 0].min(), pts[:, 1].max(), pts[:, 1].min(), pts[:, 2].max(), pts[:, 2].min())
    return pts
    # from utils.utils import save_obj
    # save_obj('../debug/volume.obj', v=pts_out.numpy())


############################torch_util######################################
def compute_rotation_matrix(angles):    # 欧拉角转旋转矩阵
    n_b = angles.size(0)
    sinx = torch.sin(angles[:, 0])
    siny = torch.sin(angles[:, 1])
    sinz = torch.sin(angles[:, 2])
    cosx = torch.cos(angles[:, 0])
    cosy = torch.cos(angles[:, 1])
    cosz = torch.cos(angles[:, 2])

    rotXYZ = torch.eye(3).view(1, 3, 3).repeat(
        n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)

    rotXYZ[0, :, 1, 1] = cosx
    rotXYZ[0, :, 1, 2] = -sinx
    rotXYZ[0, :, 2, 1] = sinx
    rotXYZ[0, :, 2, 2] = cosx
    rotXYZ[1, :, 0, 0] = cosy
    rotXYZ[1, :, 0, 2] = siny
    rotXYZ[1, :, 2, 0] = -siny
    rotXYZ[1, :, 2, 2] = cosy
    rotXYZ[2, :, 0, 0] = cosz
    rotXYZ[2, :, 0, 1] = -sinz
    rotXYZ[2, :, 1, 0] = sinz
    rotXYZ[2, :, 1, 1] = cosz

    rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

    return rotation.permute(0, 2, 1)


def normalize(x):
    if torch.linalg.norm(x) < 1e-8:
        return 0
    else:
        return x / torch.linalg.norm(x)


#################### projection ################################
def perspective_projection(pts, T, K, normalize=False, width=None, height=None):
    '''
    # pts: [Nx3]
    # T: [4x4] - Extrinsic Matrix
    # K: [3x3] - Intrinsic Matrix
    '''
    mode = 'real'
    # assert mode in ['opengl', 'k4a']
    # transform to camera view
    R, t = T[:3, :3], T[:3, 3]
    pts_camview = torch.matmul(pts.unsqueeze(-2), R.t()) + t
    # project to focal plane
    pts_camview = pts_camview * torch.tensor([1.0, 1.0, -1.0]).float().to(pts_camview.device) if mode == 'opengl' else pts_camview
    pts_camview = torch.matmul(pts_camview, K.t())
    pts_camview = pts_camview.squeeze(-2)
    # calc pixel coordinates
    pts_camview[:, 0] /= pts_camview[:, 2]
    pts_camview[:, 1] /= pts_camview[:, 2]
    pts_camview[:, 1] = height - 1 - pts_camview[:, 1] if mode=='opengl' else pts_camview[:, 1]

    # normalize pixel coordinates to [-1,1] for later grid_sample operations
    if normalize:
        pts_camview[:, 0] = pts_camview[:, 0] / (width-1) * 2 - 1   # width-1 是和grid_sample中align_corner=True相一致的，如果是align_corner=False的话，则应该 (X+0.5)/W *2 - 1  ##以上是基于默认像素值位于像素中心而非角
        pts_camview[:, 1] = pts_camview[:, 1] / (height-1) * 2 - 1  #
    return pts_camview


def projection(pts, extrs, intrs, img_w, img_h):
    '''
    :param pts: [B, N, 3]
    :param extrs: [B, V, 4, 4]
    :param intrs: [B, V, 3, 3]
    :return:    [B, V, N, 3]
    '''
    batch_num, view_num = extrs.shape[:2]
    all_pts_list = []
    # print(pts.device, center_point.device, self.extrs.device, self.intrs.device)
    for bi in range(batch_num):
        pts_list = []
        for vi in range(view_num):
            pts_camview = perspective_projection(pts[bi], extrs[bi][vi], intrs[bi][vi], normalize=True, width=img_w, height=img_h)
            pts_list.append(pts_camview)
        all_pts_list.append(torch.stack(pts_list, dim=0).unsqueeze(0))
    return torch.cat(all_pts_list, dim=0)


##################### sample ###################################
def img_feature(xy, feature, padding_mode='border'):
    # xy: [B, V, N, 2]
    # feature: [B, V, C, H, W]
    grid = xy.unsqueeze(3)  # [B, V, N, 1, 2]
    feature2d = []
    # 每个点投影到不同视角，得到特征
    for view in range(xy.shape[1]):
        pt_img_feat = torch.nn.functional.grid_sample(feature[:, view, :, :, :], grid[:, view, :, :, :], align_corners=True, padding_mode=padding_mode) # [B, C, N, 1]
        feature2d.append(pt_img_feat[:,:,:,0].unsqueeze(1))
    feature2d = torch.cat(feature2d, dim=1)  # [B, V, C, N]

    return feature2d


def sample_from_triplane_new(coordinates, feat_grid, padding_mode='zeros'):
    """重新定义各平面方向，确保满足左上角(-1, -1)，右下角（1， 1）；返回list
    coordinates: [B, N, 3] or [N, 3]
    grid: [3, C, H, W] or [3, B, C, H, W]
    return: [B, N, C, 3] or [N, C, 3]
    """
    if not coordinates.ndim == 3:
        assert coordinates.ndim == 2
        coordinates_ = coordinates.unsqueeze(0)
    else:
        coordinates_ = coordinates
    batch_num = coordinates_.shape[0]
    if feat_grid.ndim == 4:
        feat_grid_ = feat_grid.unsqueeze(1).expand(-1, batch_num, -1, -1, -1)
    else:
        assert feat_grid.ndim == 5 and feat_grid.shape[1] == batch_num
        feat_grid_ = feat_grid
    plane_num, out_feature = feat_grid.shape[0], []
    if plane_num>=1:
        xy_feature = sample_from_2dgrid(coordinates_[..., [0, 1]], feat_grid_[0], padding_mode)  # [B, N, C]
        out_feature.append(xy_feature)
    if plane_num>=2:
        zy_feature = sample_from_2dgrid(coordinates_[..., [2, 1]], feat_grid_[1], padding_mode)  # [B, N, C]
        out_feature.append(zy_feature)
    if plane_num==3:
        xz_feature = sample_from_2dgrid(coordinates_[..., [0, 2]], feat_grid_[2], padding_mode)  # [B, N, C]
        out_feature.append(xz_feature)
    if plane_num<1 or plane_num>3:
        raise NotImplementedError
    out_feature = torch.stack(out_feature, dim=-1)   # [B, N, C, 3]
    if coordinates.ndim == 2:
        out_feature = out_feature[0]

    return out_feature


def sample_from_2dgrid(coordinates, feat_grid, padding_mode='zeros', use_mine=False):
    '''
    :param feat_grid: [B, C, H, W]
    :param coordinates: [B, N, 2]
    :return: [B, N, C]
    '''
    if use_mine:
        sampled_features = my_grid_sample_2d(feat_grid, coordinates.unsqueeze(-2))
    else:
        sampled_features = torch.nn.functional.grid_sample(feat_grid, coordinates.unsqueeze(-2),
                                                           mode='bilinear', padding_mode=padding_mode, align_corners=True)  # [B, C, N, 1]
    return sampled_features[..., 0].permute(0, 2, 1)


def voxel_feature(xyz, volume_feat, padding_mode='border', use_mine=False):
    # xyz: [B, N, 3]
    # self.volume_feat:   # [B, C, D, H, W] (DHW[0, 0, 0]->[D, H, W])==>(zyx[-1, -1, -1]->[1, 1, 1])
    B, N, _ = xyz.shape
    grid = xyz.reshape(B, N, 1, 1, 3)  # [B, N, 1, 1, 3]
    if use_mine:
        feat = my_grid_sample_3d(volume_feat, grid)
    else:
        feat = torch.nn.functional.grid_sample(volume_feat, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)  # [B, C, N, 1, 1]
    return feat[:, :, :, 0, 0].permute(0, 2, 1)  # [B, N, C]


def my_grid_sample_2d(image_, points_):
    '''
    # padding_mode='zeros', mode='bilinear', align_corners=True
    :param image:  # [B, C, H, W]
    :param points: # [B, N, 1, 2]   # 左上角(-1, -1)，右下角（1， 1）
    :return: [N, 2] [B, C, N, 1]
    '''
    assert points_.shape[-2] == 1 and points_.shape[-1] == 2

    B, C, H, W = image_.shape
    out = []
    for b in range(B):
        image = image_[b].permute(2, 1, 0)  # W, H, C
        pad = 3
        image = torch.nn.functional.pad(image, pad=(0, 0, pad, pad, pad, pad), mode='constant', value=0)
        # print(image_[b].permute(2, 1, 0).shape, image.shape)
        points = points_[b].squeeze()   # N, 2

        points = (points * 0.5 + 0.5) * torch.Tensor([W - 1, H - 1]).to(points.device) + pad    # convert to image coordinate
        points.clamp_min_(1.0)
        points[:, 0].clamp_max_(W + 1. + pad)
        points[:, 1].clamp_max_(H + 1. + pad)

        l = points.to(torch.long)
        wb = points - l
        wa = 1.0 - wb

        lx, ly = l.unbind(-1)

        v00 = image[lx, ly]
        v01 = image[lx, ly + 1]
        v10 = image[lx + 1, ly]
        v11 = image[lx + 1, ly + 1]

        v0 = v00 * wa[:, 1:] + v01 * wb[:, 1:]
        v1 = v10 * wa[:, 1:] + v11 * wb[:, 1:]

        sampled = v0 * wa[:, :1] + v1 * wb[:, :1]   # [N, C]
        out.append(sampled)
    out = torch.stack(out, dim=0)   # [B, N, C]
    return out.permute(0, 2, 1).unsqueeze(-1)


def my_grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    # with torch.no_grad():
    ix_tnw = torch.floor(ix).int()
    iy_tnw = torch.floor(iy).int()
    iz_tnw = torch.floor(iz).int()

    ix_tne = ix_tnw + 1
    iy_tne = iy_tnw
    iz_tne = iz_tnw

    ix_tsw = ix_tnw
    iy_tsw = iy_tnw + 1
    iz_tsw = iz_tnw

    ix_tse = ix_tnw + 1
    iy_tse = iy_tnw + 1
    iz_tse = iz_tnw

    ix_bnw = ix_tnw
    iy_bnw = iy_tnw
    iz_bnw = iz_tnw + 1

    ix_bne = ix_tnw + 1
    iy_bne = iy_tnw
    iz_bne = iz_tnw + 1

    ix_bsw = ix_tnw
    iy_bsw = iy_tnw + 1
    iz_bsw = iz_tnw + 1

    ix_bse = ix_tnw + 1
    iy_bse = iy_tnw + 1
    iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    ix_tnw = torch.clamp(ix_tnw, 0, IW - 1)
    iy_tnw = torch.clamp(iy_tnw, 0, IH - 1)
    iz_tnw = torch.clamp(iz_tnw, 0, ID - 1)
    ix_tne = torch.clamp(ix_tne, 0, IW - 1)
    iy_tne = torch.clamp(iy_tne, 0, IH - 1)
    iz_tne = torch.clamp(iz_tne, 0, ID - 1)
    ix_tsw = torch.clamp(ix_tsw, 0, IW - 1)
    iy_tsw = torch.clamp(iy_tsw, 0, IH - 1)
    iz_tsw = torch.clamp(iz_tsw, 0, ID - 1)
    ix_tse = torch.clamp(ix_tse, 0, IW - 1)
    iy_tse = torch.clamp(iy_tse, 0, IH - 1)
    iz_tse = torch.clamp(iz_tse, 0, ID - 1)
    ix_bnw = torch.clamp(ix_bnw, 0, IW - 1)
    iy_bnw = torch.clamp(iy_bnw, 0, IH - 1)
    iz_bnw = torch.clamp(iz_bnw, 0, ID - 1)
    ix_bne = torch.clamp(ix_bne, 0, IW - 1)
    iy_bne = torch.clamp(iy_bne, 0, IH - 1)
    iz_bne = torch.clamp(iz_bne, 0, ID - 1)
    ix_bsw = torch.clamp(ix_bsw, 0, IW - 1)
    iy_bsw = torch.clamp(iy_bsw, 0, IH - 1)
    iz_bsw = torch.clamp(iz_bsw, 0, ID - 1)
    ix_bse = torch.clamp(ix_bse, 0, IW - 1)
    iy_bse = torch.clamp(iy_bse, 0, IH - 1)
    iz_bse = torch.clamp(iz_bse, 0, ID - 1)
    # with torch.no_grad():
    #     torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
    #     torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
    #     torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)
    #
    #     torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
    #     torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
    #     torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)
    #
    #     torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
    #     torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
    #     torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)
    #
    #     torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
    #     torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
    #     torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)
    #
    #     torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
    #     torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
    #     torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)
    #
    #     torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
    #     torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
    #     torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)
    #
    #     torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
    #     torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
    #     torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)
    #
    #     torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
    #     torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
    #     torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)
    tnw_val = torch.gather(
        image, 2,
        (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(
        image, 2,
        (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(
        image, 2,
        (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(
        image, 2,
        (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(
        image, 2,
        (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(
        image, 2,
        (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(
        image, 2,
        (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(
        image, 2,
        (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))
    # print(ix.shape, ix_tnw.shape, tnw_val.shape, out_val.shape)
    # print(ix[0, 0, 0], ix_tnw[0, 0, 0], tnw_val[0, 0, 0], out_val[0, 0, 0])
    # exit(0)
    return out_val