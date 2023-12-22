import cv2# .cv2 as cv2
import numpy as np
# from core.utils import get_normal_map, get_normal_map_torch
from core import get_recon_model
import os
import torch
# import core.utils as utils
from tqdm import tqdm
import json
from pytorch3d.renderer import look_at_view_transform
from core.utils import UniformBoxWarp_new, get_box_warp_param, depth2normal_ortho, get_lm_weights
from pytorch3d.renderer import TexturesVertex
from core.FaceVerseModel_v3 import get_renderer
from scipy.ndimage import gaussian_filter1d
from fit_video import render_canonical_ortho, rotate_by_theta_along_y, make_animation_transform, filter_selected_transform


def process(args, avatar_tracking_dir, calib_file=None, front_view_name=None):
    if calib_file is None:
        calib = {
            'img_res': 512,
            'intrinsics':{'0':{
                        'cam_K': np.asarray([1315.0, 0.0, 256.0, 0.0, 1315.0, 256.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3),
                        'cam_T': np.eye(4, dtype=np.float32)}}}
        front_view_name = '0'
    else:
        calib = json.loads(open(calib_file).read())

    assert os.path.exists(avatar_tracking_dir)
    avatar_tracking_dir = avatar_tracking_dir[:-1] if avatar_tracking_dir.endswith('/') else avatar_tracking_dir
    avatar_frame_ls = [int(name) for name in os.listdir(avatar_tracking_dir) if os.path.isdir(os.path.join(avatar_tracking_dir, name))]
    avatar_frame_ls.sort()
    driven_base_path = os.path.join(avatar_tracking_dir, str(avatar_frame_ls[72]))
    incre_expr = False
    smooth_coeff = False
    cam_K = np.asarray(calib["intrinsics"][front_view_name]['cam_K'], dtype=np.float32).reshape(3, 3)
    if args.audio_path is not None:
        drive_dir_name = os.path.basename(args.audio_path).split('.')[0] + ('_incre' if incre_expr else '') + ('_smooth' if smooth_coeff else '')
        animation(args, driven_base_path, drive_dir_name, audio_path=args.audio_path, smooth_coeff=smooth_coeff, incre_expr=incre_expr)
        # make_audio_transform(cam_dist=args.cam_dist, drive_base_dir=base_dir, img_res=calib['img_res'], cam_K=cam_K,
        #                      drive_save_dir=os.path.join(os.path.dirname(args.audio_path), 'audio_drive', drive_dir_name), no_pose=True,
        #                      avatar_baseframe_path=driven_base_path, drive_dir_name=drive_dir_name, view_num=1)
        make_audio_transform(cam_dist=args.cam_dist, drive_base_dir=os.path.dirname(avatar_tracking_dir), img_res=calib['img_res'], cam_K=cam_K,
                             drive_save_dir=os.path.join(os.path.dirname(args.audio_path), 'audio_drive', drive_dir_name), no_pose=True,
                             avatar_baseframe_path=driven_base_path, drive_dir_name=drive_dir_name, view_num=60)

    elif args.animation_video_tracking_dir is not None:
        args.animation_video_tracking_dir = args.animation_video_tracking_dir[:-1] if args.animation_video_tracking_dir.endswith('/') else args.animation_video_tracking_dir
        drive_base_dir = os.path.dirname(args.animation_video_tracking_dir)
        drive_dir_name = os.path.basename(os.path.dirname(avatar_tracking_dir)) + ('_incre' if incre_expr else '') + ('_smooth' if smooth_coeff else '')
        animation(args, driven_base_path, drive_dir_name, video_tracking_dir=args.animation_video_tracking_dir, smooth_coeff=smooth_coeff, incre_expr=incre_expr)

        drive_frame_ls = [int(name) for name in os.listdir(args.animation_video_tracking_dir) if os.path.isdir(os.path.join(args.animation_video_tracking_dir, name))]
        drive_frame_ls.sort()
        drive_zeropose_frameind = str(drive_frame_ls[10])

        make_animation_transform(cam_dist=args.cam_dist, drive_base_dir=drive_base_dir, drive_save_dir=args.animation_video_tracking_dir, calib=calib,
                                 cam_K=cam_K, drive_zeropose_frameind=drive_zeropose_frameind, avatar_baseframe_path=driven_base_path,
                                 drive_dir_name=drive_dir_name, view_num=45)


def animation(args, avatar_baseframe_path, drive_dir_name, audio_path=None, video_tracking_dir=None, smooth_coeff=False, incre_expr=True):
    driven_coeffs = torch.from_numpy(np.load(os.path.join(avatar_baseframe_path, 'coeffs.npy')).astype(np.float32)).unsqueeze(0).to('cuda:0')
    recon_model = get_recon_model(model=args.recon_model, device=args.device)

    scales, trans = get_box_warp_param(np.asarray([-1.5, 1.5]), np.asarray([-1.6, 1.4]), np.asarray([-1.6, 1.2]))
    orthRender_gridwarper = UniformBoxWarp_new(scales=scales, trans=trans).to('cuda:0')
    orthRender_K = [-1.0, -1.0, 0., 0.]
    orthRender_cam_param_ls = [
        {'K': orthRender_K, 'rot': recon_model.compute_rotation_matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).cuda(), 'name': 'front'},
        {'K': orthRender_K, 'rot': recon_model.compute_rotation_matrix(torch.tensor([0, -90 / 180 * np.pi, 0]).reshape(1, -1)).cuda(),
         'name': 'left'},
        {'K': orthRender_K, 'rot': recon_model.compute_rotation_matrix(torch.tensor([0, 90 / 180 * np.pi, 0]).reshape(1, -1)).cuda(),
         'name': 'right'}]
    orthRender_T = torch.zeros(1, 3, dtype=torch.float32, device='cuda:0'); orthRender_T[0, -1] = args.cam_dist
    orthRender_renderer = get_renderer(img_size=256, device='cuda:0', R=torch.eye(3, dtype=torch.float32, device='cuda:0').unsqueeze(0),
                                       T=orthRender_T, K=orthRender_K, orthoCam=True, rasterize_blur_radius=1e-6)
    orthRender = {'K': orthRender_K, 'renderer': orthRender_renderer, 'cam_param_ls': orthRender_cam_param_ls, 'gridwarper': orthRender_gridwarper}
    if audio_path is not None:
        audio_animation(audio_path, recon_model, orthRender, driven_coeffs, savedir=os.path.join(os.path.dirname(audio_path), 'audio_drive', drive_dir_name),
                        incre_expr=incre_expr, smooth_audio=smooth_coeff)
    elif video_tracking_dir is not None:
        video_animation(video_tracking_dir, recon_model, orthRender, driven_coeffs, drive_dir_name, incre_expr=incre_expr, smooth_coeff=smooth_coeff)


def video_animation(video_tracking_dir, recon_model, orthRender, driven_coeffs, drive_dir_name, smooth_coeff=False, incre_expr=True):
    name_ls = [name for name in os.listdir(video_tracking_dir) if
               (os.path.isdir(os.path.join(video_tracking_dir, name)) and os.path.exists(os.path.join(video_tracking_dir, name, 'finish')))]
    name_ls.sort()
    coeffs_seq = np.stack([np.load(os.path.join(video_tracking_dir, name, 'coeffs.npy')) for name in name_ls], axis=0)
    if smooth_coeff: coeffs_seq = gaussian_filter1d(coeffs_seq, sigma=1.0, axis=0)
    base_coeffs = torch.from_numpy(coeffs_seq[0]).unsqueeze(0).to(recon_model.device)

    for idx, name in enumerate(tqdm(name_ls)):
        coeffs = torch.from_numpy(coeffs_seq[idx]).unsqueeze(0).to(recon_model.device)

        metaFace_coeffs = driven_coeffs.clone()
        if incre_expr:
            metaFace_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims] = \
                (coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims] -
                 base_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims]) + \
                driven_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims]  # 表情分量用增量
        else:
            metaFace_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims] = \
                coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims]
        metaFace_coeffs[:, recon_model.all_dims + 33:recon_model.all_dims + 37] = coeffs[:, recon_model.all_dims + 33:recon_model.all_dims + 37]  # 迁移瞳孔
        drive_pred_dict = recon_model(metaFace_coeffs, render=True, mask_face=True)
        render_canonical_ortho(orthRender['K'], orthRender['cam_param_ls'], recon_model, orthRender['gridwarper'], drive_pred_dict['vs'],
                               orthRender['renderer'], TexturesVertex(drive_pred_dict['color']), 256, os.path.join(video_tracking_dir, name, drive_dir_name))


def audio_animation(audio_path, recon_model, orthRender, driven_coeffs, savedir, smooth_audio=False, incre_expr=True):
    audio_coeffs = np.load(audio_path)
    if smooth_audio:
        audio_coeffs = gaussian_filter1d(audio_coeffs, sigma=1.0, axis=0)

    for idx in tqdm(range(audio_coeffs.shape[0])):
        coeff = audio_coeffs[idx]
        assert len(coeff) in [171, 121]
        metaFace_coeffs = driven_coeffs.clone()
        if len(coeff) == 171:
            if incre_expr:
                metaFace_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims] += torch.from_numpy(coeff).view(1, -1).to('cuda:0')
            else:
                metaFace_coeffs[:, recon_model.id_dims:recon_model.id_dims + recon_model.exp_dims] = torch.from_numpy(coeff).view(1, -1).to('cuda:0')
        else:
            if incre_expr:
                metaFace_coeffs[:, recon_model.id_dims + 40:recon_model.id_dims + 161] += torch.from_numpy(coeff).view(1, -1).to('cuda:0')
            else:
                metaFace_coeffs[:, recon_model.id_dims + 40:recon_model.id_dims + 161] = torch.from_numpy(coeff).view(1, -1).to('cuda:0')
        drive_pred_dict = recon_model(metaFace_coeffs, render=True, mask_face=True)
        render_canonical_ortho(orthRender['K'], orthRender['cam_param_ls'], recon_model, orthRender['gridwarper'], drive_pred_dict['vs'],
                               orthRender['renderer'], TexturesVertex(drive_pred_dict['color']), 256,
                               os.path.join(savedir, str(idx)))


def make_audio_transform(cam_dist, drive_base_dir, drive_save_dir, img_res, cam_K, avatar_baseframe_path, drive_dir_name, no_pose=False,
                         drive_zero_frameind=None, view_num=60):

    tmp_T = torch.eye(4, dtype=torch.float32)
    cam_R, cam_t = look_at_view_transform(dist=cam_dist, elev=0, azim=0)
    tmp_T[:3, :3] = cam_R[0]
    tmp_T[-1, :3] = cam_t[0]

    data_dict = {'img_res': img_res, 'init_model_coeffs_path': os.path.join(avatar_baseframe_path, 'coeffs.npy')}
    data_dict['mutiview_intr_ls'] = [
        [float(cam_K[0, 0]), float(cam_K[1, 1]), float(cam_K[0, 2] / img_res), float(cam_K[1, 2] / img_res)]
        for _ in range(view_num)
    ]  # fx fy cx cy

    model0_metaFace_extr = np.load(os.path.join(avatar_baseframe_path, 'metaFace_extr.npz'))
    model0_T, model0_T_ori = model0_metaFace_extr['extr'].astype(np.float32), model0_metaFace_extr['transformation'].astype(np.float32),
    if not no_pose:
        drive_model0_head_transformation = np.load(os.path.join(drive_save_dir, drive_zero_frameind, 'metaFace_extr.npz'))['head_T'].astype(np.float32)
        drive_model0_T_ori = np.load(os.path.join(drive_save_dir, drive_zero_frameind, 'metaFace_extr.npz'))['transformation'].astype(np.float32)

    frames = []
    for fidx in os.listdir(os.path.join(drive_save_dir)):
        frame_dict = {'fidx': int(fidx)}
        frame_dict['inst_dir'] = os.path.join(drive_save_dir, fidx)
        if not no_pose:
            metaFace_extr = np.load(os.path.join(drive_save_dir, fidx, 'metaFace_extr.npz'))
            head_transformation = metaFace_extr['head_T'].astype(np.float32)
            frame_dict['head_transformation'] = (np.dot(head_transformation, np.linalg.inv(drive_model0_head_transformation))).T.tolist()  # 右乘
            model_T_ori = np.dot(np.linalg.inv(drive_model0_T_ori), metaFace_extr['transformation']).astype(np.float32)
        else:
            frame_dict['head_transformation'] = np.eye(4, dtype=np.float32).tolist()
            model_T_ori = np.eye(4, dtype=np.float32)
        # vidx, angle = 0, 0
        view_range=[0] if view_num == 1 else range(-30, 30, 60 // view_num)
        mv_info_ls = []
        for vidx, angle in enumerate(view_range):
            rot_mat = rotate_by_theta_along_y(angle / 180 * np.pi)
            camT_mesh2cam = np.dot(model0_T_ori, rot_mat)
            camT_cam2mesh = np.linalg.inv(camT_mesh2cam)
            camT_cam2mesh_ori = np.linalg.inv(np.dot(model0_T_ori, np.dot(rot_mat, model_T_ori)))
            mv_info_ls.append({
                'view_name': str(vidx),
                'transform_matrix': camT_cam2mesh.tolist(),
                'transform_matrix_ori': camT_cam2mesh_ori.tolist()
            })
        frame_dict['mutiview_info_ls'] = mv_info_ls
        frames.append(frame_dict)

    frames.sort(key=lambda x: x['fidx'])
    data_dict['frames'] = frames  # [4::8]
    jstr = json.dumps(data_dict, indent=4)

    json_name = 'drive_%s' % drive_dir_name + ('_freeview' if view_num > 1 else '')
    with open(os.path.join(drive_base_dir, json_name + '.json'), 'w') as f:
        f.write(jstr)
    if view_num > 1:
        filter_selected_transform(os.path.join(drive_base_dir, json_name + '.json'))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--avatar_tracking_dir', type=str, required=True)
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--animation_video_tracking_dir', type=str, default=None)
    parser.add_argument('--front_view_name', type=str, default=None)
    parser.add_argument('--calib_file', type=str, default=None)
    parser.add_argument('--recon_model', type=str, default='meta_simplify_v31', help='choose a 3dmm model, default: meta')
    args = parser.parse_args()
    args.cam_dist = 5.
    args.device = 'cuda:0'
    process(args, args.avatar_tracking_dir, calib_file=args.calib_file, front_view_name=args.front_view_name)