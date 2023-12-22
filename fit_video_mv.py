import cv2# .cv2 as cv2
import numpy as np
# from core.utils import get_normal_map, get_normal_map_torch
from core import get_recon_model
import os
import torch

import imutils
from tqdm import tqdm
import time
import core.losses as losses
from OpenSeeFace.tracker import Tracker
import json
from pytorch3d.renderer import look_at_view_transform
import traceback
import mediapipe as mp
from core.utils import UniformBoxWarp_new, get_box_warp_param, depth2normal_ortho, get_lm_weights
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from core.FaceVerseModel_v3 import get_renderer
from fit_video import rotate_by_theta_along_y, visualize_render_lms
import random

def process_video_mv(args, base_dir, calib_file, valid_view_name):
    use_mediapipe = True
    not_a_video = False

    calib = json.loads(open(calib_file).read())
    img_res = calib["img_res"]
    mv_img_dir = os.path.join(base_dir, 'mv_rgb%i' % img_res)
    save_dir = os.path.join(base_dir, 'video_track_singleView_v31') if len(valid_view_name) == 1 else os.path.join(base_dir, 'video_track_multiView_v31')

    fix_cam_R, fix_cam_t = look_at_view_transform(dist=args.cam_dist, elev=0, azim=0)
    view_T = np.eye(4, dtype=np.float32)
    view_T[:3, :3] = fix_cam_R[0].cpu().detach().numpy()
    view_T[:3, -1] = fix_cam_t[0].cpu().detach().numpy()

    tracker_ls, mp_tracker_ls = [], []
    for _ in valid_view_name:
        tracker_ls.append(Tracker(img_res, img_res, threshold=None, max_threads=4,
                          max_faces=4, discard_after=10, scan_every=1,
                          silent=True, model_type=4, model_dir='OpenSeeFace/models', no_gaze=False, detection_threshold=0.4,
                          use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0))
        if use_mediapipe:
            mp_tracker_ls.append(
                mp.solutions.face_mesh.FaceMesh(static_image_mode=not_a_video, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5))

    frame_ls = []
    for frame_name in os.listdir(os.path.join(mv_img_dir, valid_view_name[0])):
        if not (frame_name.split('.')[-1]=='png'):
            continue
        mvimg_info_ls = []
        all_view_exists = True
        for idx, view_name in enumerate(valid_view_name):
            if not os.path.exists(os.path.join(mv_img_dir, view_name, frame_name)):
                all_view_exists = False
                break
            mvimg_info_ls.append({
                'view_name': view_name,
                'img_path': os.path.join(mv_img_dir, view_name, frame_name),
                'cam_K': np.asarray(calib["intrinsics"][view_name]['cam_K'], dtype=np.float32).reshape(3, 3),
                'cam_T': np.asarray(calib["intrinsics"][view_name]['cam_T'], dtype=np.float32).reshape(4, 4),
            })
        if all_view_exists:
            frame_ls.append({'fidx': int(frame_name.split('.')[0]), 'res_folder': os.path.join(save_dir, frame_name.split('.')[0]),
                             'mvimg_info_ls': mvimg_info_ls})
    frame_ls.sort(key=lambda x:x['fidx'])

    valid_frame_view_dict = fit_video_(frame_ls, calib, valid_view_name, view_T, args, tracker_ls, mp_tracker_ls)
    with open(save_dir + '_valid_frame_view_dict.json', 'w') as f: f.write(json.dumps(valid_frame_view_dict, indent=4))
    make_transform(cam_dist=args.cam_dist, base_dir=base_dir, save_dir=save_dir, calib=calib, valid_view_name=valid_view_name,
                   base_zero_frameind=str(frame_ls[10]['fidx']))


def fit_video_(frame_ls, calib, valid_view_name, view_T, args, tracker_ls, mp_tracker_ls):
    sv_flag = len(valid_view_name) == 1
    rt_reg_w = 3e-1

    inner_mouth_idx = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    left_eye_idx = []#[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_idx = []#[362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    l_brow_idx = [283, 282, 295, 285, 336, 296, 334]
    r_brow_idx = [53, 52, 65, 55, 107, 66, 105]
    sideview_valid_kpsidx = inner_mouth_idx + left_eye_idx + right_eye_idx + l_brow_idx + r_brow_idx
    sideview_kps_mask = torch.tensor([i in sideview_valid_kpsidx for i in range(478)], device=args.device).float().view(1, -1, 1)

    recon_model = get_recon_model(model=args.recon_model, device=args.device)
    lm_weights = get_lm_weights(args.device, use_mediapipe=True)

    recon_model_ls = []
    resize_factor = args.tar_size / calib['img_res']
    for idx, view_name in enumerate(valid_view_name):
        cam_K = np.asarray(calib["intrinsics"][view_name]['cam_K'], dtype=np.float32).reshape(3, 3)
        cam_K[:2] = cam_K[:2] * resize_factor
        recon_model_ls.append(get_recon_model(model=args.recon_model, device=args.device, batch_size=1,
                                              img_size=args.tar_size, intr=cam_K, cam_dist=args.cam_dist))

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
    frame_ind = -1
    t0 = time.time()
    valid_frame_view_dict = {}
    for frame_dict in frame_ls:
        frame_ind += 1
        # if not frame_dict['fidx'] >= 931: continue
        print("Processing frame %d (%d / %d)" % (frame_dict['fidx'], frame_ind, len(frame_ls)), time.time()-t0, (time.time() - t0) * (len(frame_ls) - frame_ind))
        t0 = time.time()

        res_folder = frame_dict['res_folder']
        os.makedirs(res_folder, exist_ok=True)
        img_info_ls = frame_dict['mvimg_info_ls']

        valid_view_ls, view_name_ls, lms_tensor_ls, img_tensor_ls, track_crop_ls, img_arr_ls, camT_tensor_ls, cam_K_ls = [], [], [], [], [], [], [], []
        for vidx, img_info in enumerate(img_info_ls):
            view_name = img_info['view_name']

            img_path = img_info['img_path']
            ori_img_arr = cv2.imread(img_path)[:, :, ::-1]
            img_arr = ori_img_arr
            preds = tracker_ls[vidx].predict(img_arr)
            valid_view_flag = True
            if len(preds) == 0: # 若检测不到人脸，则报错
                print('No face detected!', img_path)
                valid_view_flag = False

            bbox = [0, 0, img_arr.shape[1], img_arr.shape[0]]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            face_w = bbox[2] - bbox[0]
            face_h = bbox[3] - bbox[1]
            assert face_w == face_h
            resize_factor = args.tar_size / face_h
            track_crop_ls.append({'size': (face_w, face_h), 'bbox': bbox, 'resize_factor': resize_factor})

            cam_K = np.asarray(calib["intrinsics"][view_name]['cam_K'], dtype=np.float32).reshape(3, 3)
            cam_K[0, 2] = cam_K[0, 2] - bbox[0]  # -left
            cam_K[1, 2] = cam_K[1, 2] - bbox[1]  # -top
            cam_K[:2] = cam_K[:2] * resize_factor
            recon_model_ls[vidx].set_renderer(intr=cam_K, img_size=args.tar_size, cam_dist=args.cam_dist)
            cam_K_ls.append(cam_K)

            face_img = img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))
            results = mp_tracker_ls[vidx].process(resized_face_img)
            lms = np.zeros((478, 2), dtype=np.int64)
            if results.multi_face_landmarks is None:
                valid_view_flag = False
            else:
                for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                    lms[idx, 0] = int(landmark.x * args.tar_size)
                    lms[idx, 1] = int(landmark.y * args.tar_size)

            lms_tensor_ls.append(torch.tensor(lms[np.newaxis, :, :], dtype=torch.float32, device=args.device))
            img_tensor_ls.append(torch.tensor(resized_face_img[None, ...], dtype=torch.float32, device=args.device))

            cam_T = np.asarray(calib["intrinsics"][view_name]['cam_T'], dtype=np.float32).reshape(4, 4)
            if vidx == 0:
                cam_T0 = cam_T.copy()
            camT_tensor = torch.from_numpy(np.dot(np.linalg.inv(view_T), np.dot(cam_T, np.dot(np.linalg.inv(cam_T0), view_T)))).to(args.device)
            camT_tensor_ls.append(camT_tensor)

            img_arr_ls.append(img_arr)
            view_name_ls.append(view_name)
            valid_view_ls.append(valid_view_flag)

        if (not sv_flag) and np.asarray(valid_view_ls, dtype=np.float32).sum() < 3:
            print('WARNING!', 'Too few faces detected')

        rigid_optim_params = [recon_model.get_exp_tensor(), recon_model.get_eye_tensor(), recon_model.get_rot_tensor(), recon_model.get_trans_tensor()]
        if frame_ind < 10:
            rigid_optim_params.append(recon_model.get_id_tensor())
            if not sv_flag:
                rigid_optim_params.append(recon_model.get_scale_tensor())

        num_iters_rf = 2000 if frame_ind == 0 else 200

        rigid_optimizer = torch.optim.Adam(rigid_optim_params, lr=1e-1, betas=(0.8, 0.95)) if frame_ind == 0 \
            else torch.optim.Adam(rigid_optim_params, lr=1e-2, betas=(0.5, 0.9))
        lr_rigid_optimizer = torch.optim.Adam(rigid_optim_params, lr=1e-3, betas=(0.5, 0.9))

        # # rigid optimization
        for iter_rf in tqdm(range(num_iters_rf)):
            total_loss = 0
            v_num = 0
            out_str = ''
            for vidx in range(len(valid_view_name)):
                if not valid_view_ls[vidx]:
                    continue
                pred_dict = recon_model_ls[vidx](recon_model.get_packed_tensors(), render=False, camT=camT_tensor_ls[vidx])
                if frame_ind > 0 and face_orientation_flag_ls[vidx] == 0:
                    continue
                elif frame_ind > 0 and face_orientation_flag_ls[vidx] == 1:
                    lm_loss_val = losses.lm_loss(pred_dict['lms_proj'] * sideview_kps_mask, lms_tensor_ls[vidx] * sideview_kps_mask,
                                                 lm_weights, img_size=args.tar_size, keep_dim=True)
                    lm_loss_val = (lm_loss_val.sum(1) / sideview_kps_mask.sum())[0]
                else:
                    lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms_tensor_ls[vidx], lm_weights, img_size=args.tar_size)
                total_loss += args.lm_loss_w * lm_loss_val
                out_str += '%s:%03f '%(valid_view_name[vidx], args.lm_loss_w * lm_loss_val.item())
                v_num += 1

            id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
            # print(i, total_loss.item(), id_reg_loss.item(), exp_reg_loss.item())
            total_loss = total_loss / v_num + exp_reg_loss * args.exp_reg_w + id_reg_loss * args.id_reg_w
            if frame_ind > 0:
                rt_reg_loss = (losses.get_l2(recon_model.get_rot_tensor() - rot_c) + losses.get_l2(recon_model.get_trans_tensor() - trans_c))
                total_loss += rt_reg_loss * rt_reg_w
                out_str += 'rt_reg:%03f' % (rt_reg_loss * rt_reg_w).item()

            if frame_ind > 0 and iter_rf > num_iters_rf * 0.6:
                lr_rigid_optimizer.zero_grad()
                total_loss.backward()
                lr_rigid_optimizer.step()
            else:
                rigid_optimizer.zero_grad()
                total_loss.backward()
                rigid_optimizer.step()

            with torch.no_grad(): # # zero_negExpr
                recon_model.exp_tensor[recon_model.exp_tensor < 0] *= 0

        face_orientation_ls, face_orientation_flag_ls = [], []
        for vidx in range(len(valid_view_ls)):
            pred_dict = recon_model_ls[vidx](recon_model.get_packed_tensors(), render=True, camT=camT_tensor_ls[vidx])
            lms = pred_dict['lms_t']
            lms = lms - (pred_dict['vs_t'].mean(dim=1, keepdim=True))
            face_orientation = lms[0, sideview_valid_kpsidx].mean(dim=0)
            face_orientation = face_orientation / torch.linalg.norm(face_orientation)
            if (not sv_flag) and face_orientation[-1] < 0.6:  # 太偏：完全不用，且训练时要被剔除
                face_orientation_flag_ls.append(0)
                valid_view_ls[vidx] = False
            elif (not sv_flag) and face_orientation[-1] < 0.85:
                face_orientation_flag_ls.append(1)
            else:
                face_orientation_flag_ls.append(2)
            face_orientation_ls.append(face_orientation)
        if max(face_orientation_flag_ls) < 2: face_orientation_flag_ls[0] = 2
        print(face_orientation_flag_ls, ['%.03f'%face_orientation[-1].item() for face_orientation in face_orientation_ls])
        rot_c, trans_c = recon_model.get_rot_tensor().clone().detach(), recon_model.get_trans_tensor().clone().detach()
        print('done fitting. lm_loss: %s' % out_str, 'valid_view_num', int(np.asarray(valid_view_ls, dtype=np.float32).sum()))
        valid_frame_view_dict[str(frame_dict['fidx'])] = {vn:vflag for vn, vflag in zip(valid_view_name, valid_view_ls)}
        render_lms_dict, cam_intr = {}, {}
        with torch.no_grad():
            coeffs = recon_model.get_packed_tensors()
            for idx in range(len(valid_view_name)):
                lms_tensor = lms_tensor_ls[idx]
                pred_dict = recon_model_ls[idx](coeffs, render=True, camT=camT_tensor_ls[idx], mask_face=True)
                rendered_img = torch.clip(pred_dict['rendered_img'], 0, 255).cpu().numpy().squeeze()
                proj_lms = pred_dict['lms_proj'].cpu().numpy().squeeze().astype(np.float32) / track_crop_ls[idx]['resize_factor']
                proj_lms += np.array(track_crop_ls[idx]['bbox'][:2])
                render_lms_dict[view_name_ls[idx]] = proj_lms.tolist()
                cam_intr[view_name_ls[idx]] = cam_K_ls[idx].tolist()

                if idx == 0:
                    # # save the coefficients
                    np.save(os.path.join(res_folder, 'coeffs.npy'), coeffs.detach().cpu().numpy().squeeze())
                    render_canonical_ortho(orthRender_K, orthRender_cam_param_ls, recon_model, orthRender_gridwarper, pred_dict['vs'],
                                               orthRender_renderer, TexturesVertex(pred_dict['color']), 256, res_folder)

                    split_coeffs = recon_model.split_coeffs(coeffs)
                    angles, translation, scale = split_coeffs[3], split_coeffs[5], split_coeffs[-1]

                    # # save extr
                    rotation = recon_model.compute_rotation_matrix(angles)  ###
                    cam_T, tmp_T = torch.eye(4, dtype=torch.float32).to(args.device), torch.eye(4, dtype=torch.float32).to(args.device)
                    cam_R, cam_t = look_at_view_transform(dist=args.cam_dist, elev=0, azim=0)
                    tmp_T[:3, :3] = cam_R[0]
                    tmp_T[-1, :3] = cam_t[0]

                    cam_T[:3, :3] = torch.abs(scale[0]) * torch.eye(3, dtype=torch.float32).to(args.device)
                    cam_T[-1, :3] = translation[0]
                    metaFace_extr = torch.matmul(cam_T, tmp_T).clone()  # 左乘点 P * T

                    cam_T[:3, :3] = torch.abs(scale[0]) * rotation[0]
                    cam_T[-1, :3] = translation[0]   # head_transformation
                    transformation = torch.matmul(cam_T, tmp_T)  # 左乘点 P * T

                    np.savez(os.path.join(res_folder, 'metaFace_extr'),
                             head_T=cam_T.cpu().numpy().astype(np.float32),
                             extr=metaFace_extr.cpu().numpy().astype(np.float32).T,  # 右乘
                             transformation=transformation.cpu().numpy().astype(np.float32).T,
                             self_rotation=rotation[0].cpu().numpy().astype(np.float32).T,
                             )

                # # save the composed image
                out_img = rendered_img[:, :, :3].astype(np.uint8)
                resized_out_img = cv2.resize(out_img, track_crop_ls[idx]['size'])
                composed_img = img_arr_ls[idx].copy() * 0.6 + resized_out_img * 0.4

                resized_lms = lms_tensor.cpu().detach().squeeze().numpy() / track_crop_ls[idx]['resize_factor']
                resized_lms_proj = pred_dict['lms_proj'].cpu().detach().squeeze().numpy() / track_crop_ls[idx]['resize_factor']
                composed_img = visualize_render_lms(composed_img, resized_lms, resized_lms_proj)
                ori_img = visualize_render_lms(img_arr_ls[idx].copy(), resized_lms)
                cv2.imwrite(os.path.join(res_folder, img_info_ls[idx]['view_name'] + '_ori.png'), ori_img[:, :, ::-1].astype(np.uint8))
                cv2.imwrite(os.path.join(res_folder, img_info_ls[idx]['view_name'] + '.png'), composed_img[:, :, ::-1].astype(np.uint8))

        open(os.path.join(res_folder, 'finish'), "w")
    return valid_frame_view_dict


def render_canonical_ortho(K, cam_param_ls, recon_model, gridwarper, vs_, renderer, face_color_tv, res, inst_dir):
    vs = gridwarper(vs_)
    for ridx, cam_param in enumerate(cam_param_ls):
        vs_r = torch.matmul(vs, cam_param_ls[ridx]['rot'])
        mesh = Meshes(vs_r, recon_model.tri.repeat(1, 1, 1), face_color_tv)
        rendered_img, rendered_depth = renderer(mesh)
        # utils.save_obj(os.path.join('debug', 'test_mesh.obj'), vs[0], recon_model.tri + 1, face_color.cpu().numpy().squeeze().astype(np.float32) / 255.)
        rendered_normal = depth2normal_ortho(rendered_depth.cpu()[..., 0].float(), dx=K[0] / (res // 2), dy=K[1] / (res // 2)).numpy()
        rendered_normal = ((rendered_normal[0] + 1.0) * 127.5).astype(np.uint8)
        rendered_normal[rendered_img[0, :, :, -1].cpu().numpy() == 0] = 0

        lms_t = (recon_model.get_lms(vs_r)).cpu().numpy()
        lms_proj = np.zeros((lms_t.shape[1], 2), dtype=np.float32)
        lms_proj[:, 0] = ((lms_t[0, :, 0] * -K[0] + K[2]) + 1.0) * (res // 2)
        lms_proj[:, 1] = ((lms_t[0, :, 1] * -K[1] + K[3]) + 1.0) * (res // 2)

        color_img = torch.clamp(rendered_img[0, :, :, :3], 0, 255).cpu().numpy().astype(np.uint8)
        mask_img = (color_img[..., 0:1] > 0) * (color_img[..., 1:2] > 0) * (color_img[..., 2:3] > 0)
        rendered_normal *= mask_img.astype(np.uint8)
        os.makedirs(inst_dir, exist_ok=True)
        cv2.imwrite(os.path.join(inst_dir, 'ortho_%s_normal_256_baseGama.png' % (cam_param_ls[ridx]['name'])),
                    cv2.resize(rendered_normal, dsize=(256, 256)))
        cv2.imwrite(os.path.join(inst_dir, 'ortho_%s_render_256_baseGama.png' % (cam_param_ls[ridx]['name'])),
                    cv2.resize(color_img[:, :, ::-1], dsize=(256, 256)))


def make_transform(cam_dist, base_dir, save_dir, calib, valid_view_name, base_zero_frameind):
    tmp_T = torch.eye(4, dtype=torch.float32)
    cam_R, cam_t = look_at_view_transform(dist=cam_dist, elev=0, azim=0)
    tmp_T[:3, :3] = cam_R[0]
    tmp_T[-1, :3] = cam_t[0]

    img_res = calib["img_res"]
    mv_mask_dir = os.path.join(base_dir, 'mv_mask%s' % (str(img_res)))
    mv_img_dir = os.path.join(base_dir, 'mv_rgb%s' % (str(img_res)))
    mv_bg_dir = os.path.join(base_dir, 'mv_bg%s' % (str(img_res)))

    data_dict = {'img_res': img_res}
    view_ls = []
    for idx, view_name in enumerate(valid_view_name):
        view_ls.append({
            'view_name': view_name,
            'cam_K': np.asarray(calib["intrinsics"][view_name]['cam_K'], dtype=np.float32).reshape(3, 3),
            'cam_T': np.asarray(calib["intrinsics"][view_name]['cam_T'], dtype=np.float32).reshape(4, 4),
        })
    data_dict['mutiview_intr_ls'] = [
        [float(view['cam_K'][0, 0]), float(view['cam_K'][1, 1]), float(view['cam_K'][0, 2] / img_res), float(view['cam_K'][1, 2] / img_res)]
        # fx fy cx cy
        for view in view_ls
    ]

    model0_head_transformation = np.load(os.path.join(save_dir, base_zero_frameind, 'metaFace_extr.npz'))['head_T'].astype(np.float32)
    model0_transformation = np.load(os.path.join(save_dir, base_zero_frameind, 'metaFace_extr.npz'))['transformation'].astype(np.float32)
    camT_mesh2glo = np.dot(np.linalg.inv(view_ls[0]['cam_T'].astype(np.float32)), model0_transformation).astype(np.float32)  # 右乘
    if os.path.exists(mv_bg_dir):
        data_dict['bg_path'] = [os.path.join(mv_bg_dir, '%s.png' % view) for view in valid_view_name]
    data_dict['init_model_coeffs_path'] = os.path.join(save_dir, base_zero_frameind, 'coeffs.npy')
    data_dict['base_frontal_mask_path'] = os.path.join(mv_mask_dir, valid_view_name[0], base_zero_frameind + '.png')
    frames = []
    tqdm_ls = os.listdir(os.path.join(mv_img_dir, valid_view_name[0]))
    fidx_ls = []
    for frame_name in tqdm(tqdm_ls):
        fidx = int(frame_name.split('.')[0])
        if not (fidx >= int(base_zero_frameind)):
            continue
        # if not (fidx<=1402):
        #     continue
        if not os.path.exists(os.path.join(os.path.join(save_dir, frame_name.split('.')[0]), 'finish')):
            continue
        fidx_ls.append(fidx)
        frame_dict = {'fidx': int(frame_name.split('.')[0])}
        frame_dict['inst_dir'] = os.path.join(save_dir, frame_name.split('.')[0])

        metaFace_extr = np.load(os.path.join(save_dir, frame_name.split('.')[0], 'metaFace_extr.npz'))
        head_transformation = metaFace_extr['head_T'].astype(np.float32)
        frame_dict['head_transformation'] = (np.dot(head_transformation, np.linalg.inv(model0_head_transformation))).T.tolist()  # 右乘

        camT_mesh2glo_ori = np.dot(np.linalg.inv(view_ls[0]['cam_T'].astype(np.float32)), metaFace_extr['transformation']).astype(np.float32)
        mv_info_ls = []
        for idx, view_name in enumerate(valid_view_name):
            camT_cam2mesh = np.linalg.inv(np.dot(view_ls[idx]['cam_T'].astype(np.float32), camT_mesh2glo))
            camT_cam2mesh_ori = np.linalg.inv(np.dot(view_ls[idx]['cam_T'].astype(np.float32), camT_mesh2glo_ori))
            mv_info_ls.append({
                'view_name': view_name,
                'mask_path': os.path.join(mv_mask_dir, view_name, frame_name),
                'file_path': os.path.join(mv_img_dir, view_name, frame_name),
                'transform_matrix': camT_cam2mesh.tolist(),
                'transform_matrix_ori': camT_cam2mesh_ori.tolist(),
            })
        frame_dict['mutiview_info_ls'] = mv_info_ls

        frames.append(frame_dict)

    fidx_ls.sort()
    frames.sort(key=lambda x:x['fidx'])
    random.shuffle(frames)

    all_data_dict = data_dict.copy()
    all_data_dict['frames'] = frames
    jstr = json.dumps(all_data_dict, indent=4)
    prefix = 'sv' if len(valid_view_name) == 1 else 'mv'
    with open(os.path.join(base_dir, f'{prefix}_v31_all.json'), 'w') as f:
        f.write(jstr)


def extract_video_frame(video_dir, save_dir, dst_resolution=512, skip=1, angle_t=0, crop_params_=None, only_first=False, blur_padding=False):
    def get_length(pred):
        lm = np.array(pred)
        brow_avg = (lm[19] + lm[24]) * 0.5
        bottom = lm[8]
        length = np.sqrt(np.sum(np.square(brow_avg - bottom)))
        return length * 1.05

    def pad_blur(img, pad, blur=None):
        # pad [left, up, right, bottom]
        from scipy.ndimage import gaussian_filter
        if blur is None: blur = (sum(pad) / 4) / 0.3 * 0.02 * 0.1
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        low_res = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        # blur = qsize * 0.02 * 0.1
        low_res = gaussian_filter(low_res, [blur, blur, 0])
        low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        median = np.median(median, axis=(0, 1))
        img += (median - img) * np.clip(mask, 0.0, 1.0)
        out_img = np.uint8(np.clip(np.rint(img), 0, 255))
        return out_img

    crop_params = {} if crop_params_ is None else crop_params_
    assert isinstance(crop_params, dict)
    for video_name in os.listdir(video_dir):
        if not video_name.endswith('mp4'): continue
        view_name = video_name.split('.')[0]
        video_path = os.path.join(video_dir, video_name)
        dst_save_dir = os.path.join(save_dir, view_name)
        videoCapture = cv2.VideoCapture(video_path)
        os.makedirs(dst_save_dir, exist_ok=True)
        # 获得码率及尺寸
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(video_name, size, fNUMS, fps)

        # First Frame for crop params
        flag, frame = videoCapture.read()
        frame = imutils.rotate_bound(frame, angle_t)
        if flag is False:
            print('error reading the video file %s' % video_path)
            return
        # cv2.imwrite(os.path.join(dst_save_dir, '-1.png'), frame)
        if crop_params_ is not None:
            top, left, resolution, pad = crop_params[view_name]
            if only_first:
                frame = pad_blur(frame, pad=(pad, pad, pad, pad)) if blur_padding else cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                cv2.imwrite(os.path.join(dst_save_dir, '0.png'),
                            cv2.resize(frame[top: top + resolution, left: left + resolution], dsize=(dst_resolution, dst_resolution), interpolation=cv2.INTER_LINEAR))
        else:
            orig_h, orig_w = frame.shape[:2]
            tracker = Tracker(orig_w, orig_h, threshold=None, max_threads=1,
                              max_faces=1, discard_after=10, scan_every=1,
                              silent=True, model_type=4, model_dir='OpenSeeFace/models', no_gaze=True, detection_threshold=0.6,
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=1)
            preds = tracker.predict(frame[:, :, ::-1])
            if len(preds) == 0:
                print('no face detected')
                cv2.imwrite(os.path.join(dst_save_dir, '-1.png'), frame)
                exit(0)

            lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)[:, [1, 0]]
            # 向外扩展图像防止crop时超出边界
            border = 0#min(frame.shape[:2]) // 4
            # crop图像为正方形时的半边长
            length_in = int(get_length(lms))
            # crop图像为正方形时的中心点
            center = lms[29].copy()
            center += np.array((0, 0), np.int64)
            center = (center + border).astype(np.int64)
            top = center[1] - length_in
            left = center[0] - length_in
            resolution = 2 * length_in
            pad = border
            bottom, right = top + 2 * length_in, left + 2 * length_in
            frame = pad_blur(frame, pad=(border, border, border, border)) if blur_padding else cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            cv2.imwrite(os.path.join(dst_save_dir, '0.png'),
                        cv2.resize(frame[top: bottom, left: right], dsize=(dst_resolution, dst_resolution), interpolation=cv2.INTER_LINEAR))
            crop_params[view_name] = [top, left, resolution, pad]

        if not only_first:
            # Read and Crop
            print('angle_t, top, left, resolution, pad:', angle_t, top, left, resolution, pad)
            bottom, right = top + resolution, left + resolution
            print('Extracting frames from video %s' % video_path)
            for count in tqdm(range(1, fNUMS+1)):
                flag, frame = videoCapture.read()
                if not flag:
                    break
                if skip > 1 and not (count % skip == 0):
                    continue
                frame = imutils.rotate_bound(frame, angle_t)
                if pad > 0:
                    frame = pad_blur(frame, pad=(pad, pad, pad, pad)) if blur_padding else cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                crop_frame = frame[top: bottom, left: right]

                cv2.imwrite(os.path.join(dst_save_dir, str(count) + '.png'),
                            cv2.resize(crop_frame, dsize=(dst_resolution, dst_resolution), interpolation=cv2.INTER_LINEAR))
            videoCapture.release()

    return crop_params


def Bg_Matting(root_dir, save_dir, bg_dir=None):
    if bg_dir is None:
        model = torch.jit.load('BgMatting_models/rvm_resnet50_fp32.torchscript').to('cuda')
        model = torch.jit.freeze(model)
        print('Background matting for %s' % root_dir)
        for view_idx in os.listdir(root_dir):
            img_dir = os.path.join(root_dir, view_idx)
            sort_idx = [int(dir_name.split('.')[0]) for dir_name in os.listdir(img_dir)]
            rec = [None] * 4  # 初始值设置为 None
            count = 0
            t0 = time.time()
            os.makedirs(os.path.join(save_dir, view_idx), exist_ok=True)
            for idx, img_name in tqdm(sorted(zip(sort_idx, os.listdir(img_dir)), key=lambda x: x[0])):
                src = cv2.imread(os.path.join(img_dir, img_name))
                src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                src = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0).to('cuda')
                fgr, pha, *rec = model(src, *rec, downsample_ratio=1.0)
                mask = (pha[0, 0] * 255).cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, view_idx, img_name), mask)
                count += 1
                # print(count, (time.time() - t0) / count)
    else:
        from torchvision import transforms
        from torchvision.datasets import DatasetFolder
        from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
        from torch.utils.data import DataLoader
        class ImageFolder(DatasetFolder):
            def __init__( self, root: str, transform=None, loader=default_loader, is_valid_file=None, target_transform=None):
                super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file)

            def __getitem__(self, index: int):
                path, target = self.samples[index]
                frame_idx = int(os.path.basename(path).split('.')[0])
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return sample, target, frame_idx

        device = torch.device('cuda')
        precision = torch.float32

        input_res = 512
        test_transform = transforms.Compose([
            transforms.Resize((input_res, input_res)),
            transforms.ToTensor()]
        )

        bg_imgs_ls = dict()
        res_ls = dict()
        for name in os.listdir(bg_dir):
            bg_img = cv2.imread(os.path.join(bg_dir, name, '0.png'))
            res_ls[name.split('.')[0]] = bg_img.shape[0]
            bg_img = cv2.resize(bg_img, dsize=(input_res, input_res))
            bg_img = torch.from_numpy(bg_img.astype(np.float32) / 255).permute(2, 0, 1)
            bg_imgs_ls[name.split('.')[0]] = bg_img.to(precision).to(device)

        test_dataset = ImageFolder(root_dir, transform=test_transform)
        class_to_idx = test_dataset.class_to_idx
        camera_bg = [None for i in class_to_idx.keys()]
        save_dir_ls = [None for i in class_to_idx.keys()]
        out_res = [None for i in class_to_idx.keys()]
        for class_name in class_to_idx.keys():
            camera_bg[class_to_idx[class_name]] = bg_imgs_ls[class_name]
            save_dir_ls[class_to_idx[class_name]] = os.path.join(save_dir, class_name)
            out_res[class_to_idx[class_name]] = res_ls[class_name]
            os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=16)

        model = torch.jit.load('/media/zxc/hdd2/Code/BackGroundMatting/pretrained_models/torchscript_resnet101_fp32.pth')
        model.backbone_scale = 0.25  # * (input_res // 1024)
        model.refine_mode = 'sampling'
        model.refine_sample_pixels = 80_000

        model = model.to(device)

        for test_data in tqdm(test_dataloader):
            imgs, camera_idx, frame_idx = test_data
            bg_imgs, save_paths, save_res = [], [], []
            for i in range(camera_idx.shape[0]):
                bg_imgs.append(camera_bg[camera_idx[i]])
                save_paths.append(os.path.join(save_dir_ls[camera_idx[i]], str(int(frame_idx[i])) + '.png'))
                save_res.append(out_res[camera_idx[i]])

            src = imgs.to(precision).to(device)
            bgr = torch.stack(bg_imgs, 0)
            pha, fgr = model(src, bgr)[:2]

            for i in range(pha.shape[0]):
                mask = (pha[i, 0] * 255).cpu().numpy().astype(np.uint8)
                cv2.imwrite(save_paths[i], mask)


def make_calib(calib_file, base_dir, crop_params, dst_resolution):
    def calculate_new_intrinsic(intr, mode, param):
        '''
        mode:   resize,     crop,   padding
        param:  (fx, fy),     (left, top)
        '''
        cam_K = intr.copy()
        if mode == 'resize':
            cam_K[0] *= param[0]
            cam_K[1] *= param[1]
        elif mode == 'crop':
            cam_K[0, 2] = cam_K[0, 2] - param[0]  # -left
            cam_K[1, 2] = cam_K[1, 2] - param[1]  # -top
        elif mode == 'padding':
            cam_K[0, 2] = cam_K[0, 2] + param[0]  # + padding left
            cam_K[1, 2] = cam_K[1, 2] + param[1]  # + padding top
        else:
            assert False
        return cam_K

    calib = json.loads(open(calib_file).read())
    save_calib = {'img_res': dst_resolution, 'intrinsics': dict()}
    for cam_name in crop_params.keys():
        top, left, resolution, pad = crop_params[cam_name]
        cam_K = np.asarray(calib[cam_name]['K'], dtype=np.float32).reshape(3, 3)
        # if cam_name == '21334221':
        #     print(cam_K)
        cam_K = calculate_new_intrinsic(cam_K, mode='padding', param=(pad, pad))
        cam_K = calculate_new_intrinsic(cam_K, mode='crop', param=(left, top))
        # if cam_name == '21334221':
        #     print(cam_K)
        cam_K = calculate_new_intrinsic(cam_K, mode='resize', param=(dst_resolution / resolution, dst_resolution / resolution))
        # if cam_name == '21334221':
        #     print(cam_K)
        cam_T = np.eye(4, dtype=np.float32)
        cam_T[:3, :3] = np.asarray(calib[cam_name]['R'], dtype=np.float32).reshape(3, 3)
        cam_T[:3, 3:] = np.asarray(calib[cam_name]['T'], dtype=np.float32).reshape(3, 1)
        save_calib['intrinsics'][cam_name] = {'cam_K': cam_K.reshape(-1).tolist(), 'cam_T': cam_T.reshape(-1).tolist()}

    jstr = json.dumps(save_calib, indent=4)
    with open(os.path.join(base_dir, 'calib_%d.json' % dst_resolution), 'w') as f_obj:
        f_obj.write(jstr)


def preprocess_dataset(video_dir, bg_dir, base_dir, calib_file, resolution, crop_file, skip, angle_t):
    crop_params = None if crop_file is None else json.loads(open(crop_file).read())
    crop_params = extract_video_frame(video_dir, os.path.join(base_dir, 'mv_rgb%d' % resolution), resolution, skip, angle_t, crop_params)   # mv_rgb & crop_param
    jstr = json.dumps(crop_params, indent=4)
    with open(os.path.join(base_dir, 'crop_param.json'), 'w') as f:
        f.write(jstr)
    print('Done! The crop params are saved in %s' % os.path.join(base_dir, 'crop_param.json'))

    if bg_dir is None:
        Bg_Matting(os.path.join(base_dir, 'mv_rgb%i' % resolution), os.path.join(args.base_dir, 'mv_mask%i' % resolution))
    else:
        extract_video_frame(bg_dir, os.path.join(base_dir, 'mv_bg%i' % resolution), resolution, skip, angle_t, crop_params, only_first=True)   # mv_bg
        Bg_Matting(os.path.join(base_dir, 'mv_rgb%i' % resolution), os.path.join(args.base_dir, 'mv_mask%i' % resolution), os.path.join(args.base_dir, 'mv_bg%i' % resolution))
    make_calib(calib_file, base_dir, crop_params, resolution)            # calib

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--bg_path', type=str, required=True)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--calib_file', type=str, required=True)
    parser.add_argument('--crop_file', type=str, default=None)
    parser.add_argument('--view_name', type=str, nargs='+')
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--angle', type=int, default=0)
    parser.add_argument('--tar_size', type=int, default=512, help='size for rendering window. We use a square window.')
    parser.add_argument('--recon_model', type=str, default='meta', help='choose a 3dmm model, default: meta')
    parser.add_argument('--lm_loss_w', type=float, default=1e3, help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1e-2, help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-3, help='weight for id coefficient regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=1e-3,    # 8e-3
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-5, help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1, help='weight for texture reflectance loss.')
    args = parser.parse_args()
    args.cam_dist = 5.
    args.device = 'cuda:0'
    args.recon_model = 'meta_simplify_v31'

    # preprocess_dataset(args.video_path, args.bg_path, args.base_dir, args.calib_file, args.tar_size, args.crop_file, args.skip, args.angle)
    process_video_mv(args, args.base_dir, os.path.join(args.base_dir, 'calib_%d.json' % args.tar_size), args.view_name)