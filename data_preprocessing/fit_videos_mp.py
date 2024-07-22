import cv2
import numpy as np
import multiprocessing
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import time
import core.losses as losses
from OpenSeeFace.tracker import Tracker
import json
from pytorch3d.renderer import look_at_view_transform
import traceback
import mediapipe as mp
from datetime import datetime
count = multiprocessing.Value('i', 0)   # multiprocessing.Value对象和Process一起使用的时候，可以像上面那样作为全局变量使用，也可以作为传入参数使用。但是和Pool一起使用的时候，只能作为全局变量使用
total = multiprocessing.Value('i', 0)


def fit_faceverse(args):
    focal_ratio = args.focal_ratio
    num_thread = 8
    base_dir = args.base_dir
    assert os.path.exists(base_dir)
    if args.save_lmscounter or args.save_fvmask: assert 'images512x512' in base_dir  # 此处主要是因为之后保存fvmask和lmscounter时是用的replace
    save_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'fv_tracking') if args.save_dir is None else args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    img_res = 512
    skip = args.skip
    cam_K = np.eye(3, dtype=np.float32)
    cam_K[0, 0] = cam_K[1, 1] = focal_ratio * img_res
    cam_K[0, 2] = cam_K[1, 2] = img_res // 2

    all_frames = 0
    sub_class_ls = []
    sub_classes = [sub_class for sub_class in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, sub_class))]
    if not args.trick == 0:
        assert args.trick in [-1, 1]
        sub_classes = sub_classes[::2] if args.trick == 1 else sub_classes[1::2]
    for sub_class in tqdm(sub_classes):
        sub_dir = os.path.join(base_dir, sub_class)
        if not os.path.isdir(sub_dir):
            continue
        frame_ls = []
        for img_name in os.listdir(sub_dir):
            if not img_name.endswith('png'): continue
            res_folder = os.path.join(sub_dir.replace(base_dir, save_dir), img_name.split('.')[0])
            if skip and os.path.exists(os.path.join(res_folder, 'finish')):
                continue
            frame_ls.append({'img_path': os.path.join(sub_dir, img_name),
                             'save_dir': res_folder})
        if len(frame_ls) == 0: continue
        frame_ls.sort(key=lambda x: int(os.path.basename(x['img_path']).split('.')[0].split('_')[-1]))
        sub_class_ls.append({'video_name': sub_class, 'frame_ls': frame_ls})
        all_frames += len(frame_ls)

    total.value = all_frames
    num_thread = min(num_thread, len(sub_class_ls))
    print('base_dir:', base_dir)
    print('save_dir:', save_dir)
    print('skip:', skip); print('num_thread:', num_thread)
    print('total', total.value)
    if num_thread > 1:
        p = multiprocessing.Pool(num_thread)
        num_videos = len(sub_class_ls)
        all_list = [sub_class_ls[i * (num_videos // num_thread): (i + 1) * (num_videos // num_thread)] for i in range(num_thread)] + \
                   [sub_class_ls[num_thread * (num_videos // num_thread):]]
        data_ls = [{'img_res': img_res, 'video_ls': ls, 'save_dir': save_dir, 'cam_K': cam_K, 'save_fvmask': args.save_fvmask, 'save_lmscounter': args.save_lmscounter} for ls in all_list]
        p.map(fit_videos_, data_ls)

        p.close()
        p.join()
    else:
        fit_videos_({'img_res': img_res, 'video_ls': sub_class_ls, 'save_dir': save_dir, 'cam_K': cam_K, 'save_fvmask': args.save_fvmask, 'save_lmscounter': args.save_lmscounter})

    no_face_log = []
    for name in os.listdir(save_dir):
        if name.endswith('no_face_log.json'):
            no_face_log += json.loads(open(os.path.join(save_dir, name)).read())
    if len(no_face_log) > 0:
        jstr = json.dumps(no_face_log, indent=4)
        with open(os.path.join(save_dir, str(datetime.now()) + '_total_no_face_log.json'), 'w') as f:
            f.write(jstr)


def fit_videos_(data):
    img_res, video_ls, save_dir, cam_K, save_fvmask, save_lmscounter = data['img_res'], data['video_ls'], data['save_dir'], data['cam_K'], data['save_fvmask'], data['save_lmscounter']
    print('Fitting %d Videos' % len(video_ls))
    cam_K[:2] = cam_K[:2] * (args.tar_size / img_res)

    tracker = Tracker(img_res, img_res, threshold=None, max_threads=4,
                      max_faces=4, discard_after=10, scan_every=1,
                      silent=True, model_type=4, model_dir='OpenSeeFace/models', no_gaze=False, detection_threshold=0.4,
                      use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)
    mp_tracker = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    recon_model = get_recon_model(model=args.recon_model, device=args.device, batch_size=1,
                                  img_size=args.tar_size, intr=cam_K, cam_dist=args.cam_dist)
    no_face_log = []
    for vidx, video_info in enumerate(video_ls):
        print(video_info['frame_ls'][0]['img_path'], vidx)
        no_face_log_ = fit_(video_info['frame_ls'], recon_model, img_res, args, tracker, mp_tracker, cont_opt=False,
                            first_video=True if vidx==0 else False, reg_RT=True, save_fvmask=save_fvmask, save_lmscounter=save_lmscounter)
        if len(no_face_log_) == 0: open(os.path.join(save_dir, video_info['video_name'], 'finish'), "w")
        else:
            if no_face_log_[0][0] == 'LargeRot': open(os.path.join(save_dir, video_info['video_name'], 'LargeRot'), "w")
            if no_face_log_[0][0] == 'NoFace': open(os.path.join(save_dir, video_info['video_name'], 'NoFace'), "w")
            if no_face_log_[0][0] == 'SamllFace': open(os.path.join(save_dir, video_info['video_name'], 'SamllFace'), "w")
        no_face_log += no_face_log_
    if len(no_face_log) > 0:
        jstr = json.dumps(no_face_log, indent=4)
        with open(os.path.join(save_dir, str(datetime.now()) + '_no_face_log.json'), 'w') as f:
            f.write(jstr)
    else:
        print('no_face_log is zero')


# 默认固定相机内参, 固定camera,且不支持优化pts_disp
def fit_(frame_ls, recon_model, img_res, args, tracker, mp_tracker, first_video=False, save_mesh=False, keep_id=True, reg_RT=False,
         save_fvmask=None, save_lmscounter=None, cont_opt=False):
    lm_weights = utils.get_lm_weights(args.device, use_mediapipe=True)
    resize_factor = args.tar_size / img_res

    rt_reg_w = 0.1 if reg_RT else 0.
    num_iters_rf = 100 if keep_id else 500

    frame_ind = 0
    no_face_log = []

    for frame_dict in frame_ls:
        frame_ind += 1
        res_folder = frame_dict['save_dir']
        os.makedirs(res_folder, exist_ok=True)

        img_path = frame_dict['img_path']
        with count.get_lock():
            count.value += 1
            print('(%d / %d) Processing frame %s, first_video=%d' % (count.value, total.value, img_path, int(first_video)))
        img_arr = cv2.imread(img_path)[:, :, ::-1]

        preds = tracker.predict(img_arr)
        if len(preds) == 0: # Stop processing the video, if no face is detected
            print('No face detected!', img_path)
            no_face_log.append(['NoFace', img_path])
            break

        resized_face_img = cv2.resize(img_arr, (args.tar_size, args.tar_size))

        results = mp_tracker.process(resized_face_img)
        if results.multi_face_landmarks is None:
            print('No face detected!', img_path)
            no_face_log.append(['NoFace', img_path])
            break

        lms = np.zeros((478, 2), dtype=np.int64)
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            lms[idx, 0] = int(landmark.x * args.tar_size)
            lms[idx, 1] = int(landmark.y * args.tar_size)

        if max(max(lms[:, 0])-min(lms[:, 0]), max(lms[:, 1])-min(lms[:, 1])) < args.tar_size / 3:
            print('Too small face detected!', img_path)
            no_face_log.append(['SamllFace', img_path])
            break


        lms_tensor = torch.tensor(lms[np.newaxis, :, :], dtype=torch.float32, device=args.device)

        if cont_opt:
            if os.path.exists(os.path.join(res_folder, 'coeffs.npy')):
                coeffs = torch.from_numpy(np.load(os.path.join(res_folder, 'coeffs.npy'))).unsqueeze(0).cuda()
                id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeffs)
                recon_model.init_coeff_tensors(id_coeff=id_coeff, tex_coeff=tex_coeff, exp_coeff=exp_coeff, gamma_coeff=gamma, trans_coeff=translation,
                                               rot_coeff=angles, scale_coeff=scale, eye_coeff=eye_coeff)
                first_video = False

        if keep_id and frame_ind > 1:
            rigid_optim_params = [recon_model.get_rot_tensor(), recon_model.get_trans_tensor(), recon_model.get_exp_tensor(),
                                  recon_model.get_eye_tensor()]
        else:
            rigid_optim_params = [recon_model.get_rot_tensor(), recon_model.get_trans_tensor(), recon_model.get_exp_tensor(),
                                  recon_model.get_eye_tensor(), recon_model.get_id_tensor()]
        rigid_optimizer = torch.optim.Adam(rigid_optim_params, lr=5e-2 if (first_video and frame_ind == 1) else 1e-2, betas=(0.8, 0.95))
        lr_rigid_optimizer = torch.optim.Adam(rigid_optim_params, lr=1e-3, betas=(0.5, 0.9))

        # # rigid optimization
        num_iters = 5 * num_iters_rf if (keep_id and frame_ind == 1) else num_iters_rf
        if first_video and frame_ind == 1: num_iters *= 5
        for iter_rf in range(num_iters * 5):
            pred_dict = recon_model(recon_model.get_packed_tensors(), render=False)
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms_tensor, lm_weights, img_size=args.tar_size)
            if iter_rf > num_iters and lm_loss_val.item() < 5e-5: break

            id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())

            total_loss = args.lm_loss_w * lm_loss_val + exp_reg_loss * args.exp_reg_w + id_reg_loss * args.id_reg_w
            if frame_ind > 1:
                rt_reg_loss = losses.get_l2(recon_model.get_rot_tensor() - rot_c) + losses.get_l2(recon_model.get_trans_tensor() - trans_c)
                total_loss += rt_reg_loss * rt_reg_w

            if frame_ind > 1 and iter_rf > num_iters * 0.6:
                lr_rigid_optimizer.zero_grad()
                total_loss.backward()
                lr_rigid_optimizer.step()
            else:
                rigid_optimizer.zero_grad()
                total_loss.backward()
                rigid_optimizer.step()

            # # zero_negExpr
            with torch.no_grad():
                recon_model.exp_tensor[recon_model.exp_tensor < 0] *= 0

        rot_c, trans_c = recon_model.get_rot_tensor().clone().detach(), recon_model.get_trans_tensor().clone().detach()
        with torch.no_grad():
            coeffs = recon_model.get_packed_tensors()

            pred_dict = recon_model(coeffs, render=True, mask_face=True)
            rendered_img = torch.clip(pred_dict['rendered_img'], 0, 255).cpu().numpy().squeeze()

            out_img = rendered_img[:, :, :3].astype(np.uint8)
            resized_out_img = cv2.resize(out_img, (img_res, img_res))

            # # save the coefficients
            np.save(os.path.join(res_folder, 'coeffs.npy'), coeffs.detach().cpu().numpy().squeeze())

            split_coeffs = recon_model.split_coeffs(coeffs)
            tex_coeff, angles, translation, scale = split_coeffs[2], split_coeffs[3], split_coeffs[5], split_coeffs[-1]
            # # save the mesh into obj format
            if save_mesh:
                vs = pred_dict['vs'].cpu().numpy().squeeze()
                tri = pred_dict['tri'].cpu().numpy().squeeze()

                color = torch.clip(recon_model.get_color(tex_coeff), 0, 255).cpu().numpy().squeeze().astype(np.float32) / 255
                utils.save_obj(os.path.join(res_folder, 'mesh.obj'), vs, tri + 1, color)

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
            cam_T[-1, :3] = translation[0]
            transformation = torch.matmul(cam_T, tmp_T)  # 左乘点 P * T

            np.savez(os.path.join(res_folder, 'metaFace_extr'),
                     extr=metaFace_extr.cpu().numpy().astype(np.float32).T,  # 右乘
                     transformation=transformation.cpu().numpy().astype(np.float32).T,
                     self_rotation=rotation[0].cpu().numpy().astype(np.float32).T,
                     self_scale=scale[0].cpu().numpy().astype(np.float32),
                     self_translation=translation[0].cpu().numpy().astype(np.float32),
                     self_angle=angles[0].cpu().numpy().astype(np.float32))

            # # save the composed image
            composed_img = img_arr * 0.6 + resized_out_img * 0.4
            resized_lms = lms_tensor.cpu().detach().squeeze().numpy() / resize_factor
            resized_lms_proj = pred_dict['lms_proj'].cpu().detach().squeeze().numpy() / resize_factor
            composed_img = visualize_render_lms(composed_img, resized_lms, resized_lms_proj)
            cv2.imwrite(os.path.join(res_folder, 'composed_render.png'), composed_img[:, :, ::-1].astype(np.uint8))

            if save_fvmask is not None:
                out_mask = (np.linalg.norm(resized_out_img, axis=-1) > 0).astype(np.float32) * 255
                os.makedirs(os.path.dirname(img_path.replace('images512x512', save_fvmask)), exist_ok=True)
                cv2.imwrite(img_path.replace('images512x512', save_fvmask), out_mask.astype(np.uint8))

            if save_lmscounter is not None:
                lms_proj = pred_dict['lms_proj'].cpu().detach().squeeze().numpy()
                black_img = np.zeros((args.tar_size, args.tar_size, 3), dtype=np.uint8)
                draw_img = draw_lms_counter(black_img, lms_proj)
                os.makedirs(os.path.dirname(img_path.replace('images512x512', save_lmscounter)), exist_ok=True)
                cv2.imwrite(img_path.replace('images512x512', save_lmscounter), draw_img)

        open(os.path.join(res_folder, 'finish'), "w")

    return no_face_log


def visualize_render_lms(composed_img, resized_lms, resized_lms_proj):
    for i in range(resized_lms.shape[0]):
        cv2.circle(composed_img,
                   (round(resized_lms[i, 0]), round(resized_lms[i, 1])), 1,
                   (255, 0, 0), -1)
    for i in [0, 8, 16, 20, 24, 30, 47, 58, 62]:
        cv2.putText(composed_img, str(i),
                    (round(resized_lms[i, 0]), round(resized_lms[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

    for i in range(resized_lms.shape[0]):
        cv2.circle(composed_img,
                   (round(resized_lms_proj[i, 0]), round(resized_lms_proj[i, 1])), 1,
                   (0, 255, 0), -1)
    for i in [0, 8, 16, 20, 24, 30, 47, 58, 62]:
        cv2.putText(composed_img, str(i),
                    (round(resized_lms_proj[i, 0]), round(resized_lms_proj[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    return composed_img


def draw_lms_counter(img, lms_proj):
    lms_proj_coords = np.round(lms_proj).astype(np.int32)
    outter_mouth_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 76, 185, 40, 39, 37]
    inner_mouth_idx = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    l_brow_idx = [283, 282, 295, 285, 336, 296, 334]
    r_brow_idx = [53, 52, 65, 55, 107, 66, 105]
    draw_img = cv2.polylines(img.copy(), [lms_proj_coords[outter_mouth_idx]], True, (255, 0, 0), 4)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[inner_mouth_idx]], True, (255, 0, 0), 4)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[left_eye_idx]], True, (0, 255, 0), 2)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[right_eye_idx]], True, (0, 255, 0), 2)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[l_brow_idx]], True, (0, 255, 0), 2)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[r_brow_idx]], True, (0, 255, 0), 2)
    draw_img = cv2.circle(draw_img, (lms_proj_coords[473, 0], lms_proj_coords[473, 1]), 4, [0, 0, 255], -1)
    draw_img = cv2.circle(draw_img, (lms_proj_coords[468, 0], lms_proj_coords[468, 1]), 4, [0, 0, 255], -1)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[474:478]], True, (0, 255, 0), 1)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[469:473]], True, (0, 255, 0), 1)

    return draw_img


def gen_mouth_mask(lms_2d, new_crop=True):
    lm = lms_2d[np.newaxis, ...]
    # # lm: (B, 68, 2) [-1, 1]
    if new_crop:
        lm_mouth_outer = lm[:, [164, 18, 57, 287]]  # up, bottom, left, right corners and others
        # lm_mouth_outer = lm[:, [2, 200, 212, 432]]  # up, bottom, left, right corners and others
        mouth_mask = np.concatenate([np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True),
                                     np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True),
                                     np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True),
                                     np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True)], 1)  # (B, 4)
    else:
        lm_mouth_outer = lm[:, [0, 17, 61, 291, 39, 269, 405, 181]]  # up, bottom, left, right corners and others
        mouth_avg = np.mean(lm_mouth_outer, axis=1, keepdims=False)  # (B, 2)
        ups, bottoms = np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True), np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True)
        lefts, rights = np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True), np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True)
        mask_res = np.max(np.concatenate((ups - bottoms, rights - lefts), axis=1), axis=1, keepdims=True) * 1.2
        mask_res = mask_res.astype(int)
        mouth_mask = np.concatenate([(mouth_avg[:, 1:] - mask_res // 2).astype(int),
                                     (mouth_avg[:, 1:] + mask_res // 2).astype(int),
                                     (mouth_avg[:, 0:1] - mask_res // 2).astype(int),
                                     (mouth_avg[:, 0:1] + mask_res // 2).astype(int)], 1)  # (B, 4)
    return mouth_mask[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--tar_size', type=int, default=512, help='size for rendering window. We use a square window.')
    parser.add_argument('--recon_model', type=str, default='meta', help='choose a 3dmm model, default: meta')
    parser.add_argument('--lm_loss_w', type=float, default=1e3, help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1e-2, help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-3, help='weight for id coefficient regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=1e-3,    # 8e-3
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-5, help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1, help='weight for texture reflectance loss.')
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--save_fvmask', type=str, default=None)
    parser.add_argument('--save_lmscounter', type=str, default=None)
    parser.add_argument('--num_threads', default=8)
    parser.add_argument('--trick', type=int, default=0)
    args = parser.parse_args()
    args.focal_ratio = 4.2647  # the focal used by EG3D
    args.cam_dist = 5.
    args.device = 'cuda:0'
    args.recon_model = 'meta_simplify_v31'
    fit_faceverse(args)
