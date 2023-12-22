import argparse
import numpy as np

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--tar_size', type=int, default=512,
                                 help='size for rendering window. We use a square window.')
        self.parser.add_argument('--padding_ratio', type=float, default=1.0,
                                 help='enlarge the face detection bbox by a margin.')
        self.parser.add_argument('--recon_model', type=str, default='meta',
                                 help='choose a 3dmm model, default: meta')
        self.parser.add_argument('--first_rf_iters', type=int, default=1000,
                                 help='iteration number of rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--first_nrf_iters', type=int, default=500,
                                 help='iteration number of non-rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--rest_rf_iters', type=int, default=80, #100, #50,
                                 help='iteration number of rigid fitting for the remaining frames in video fitting.')
        self.parser.add_argument('--rest_nrf_iters', type=int, default=30, #60, #30,
                                 help='iteration number of non-rigid fitting for the remaining frames in video fitting.')
        self.parser.add_argument('--rf_lr', type=float, default=1e-2,
                                 help='learning rate for rigid fitting')
        self.parser.add_argument('--nrf_lr', type=float, default=1e-2,
                                 help='learning rate for non-rigid fitting')
        self.parser.add_argument('--lm_loss_w', type=float, default=3e3,
                                 help='weight for landmark loss')
        self.parser.add_argument('--rgb_loss_w', type=float, default=1.6,
                                 help='weight for rgb loss')
        self.parser.add_argument('--id_reg_w', type=float, default=1e-3,
                                 help='weight for id coefficient regularizer')
        self.parser.add_argument('--exp_reg_w', type=float, default=1.5e-4,
                                 help='weight for expression coefficient regularizer')
        self.parser.add_argument('--tex_reg_w', type=float, default=3e-4,
                                 help='weight for texture coefficient regularizer')
        self.parser.add_argument('--rot_reg_w', type=float, default=1,
                                 help='weight for rotation regularizer')
        self.parser.add_argument('--trans_reg_w', type=float, default=1,
                                 help='weight for translation regularizer')

        self.parser.add_argument('--tex_w', type=float, default=1,
                                 help='weight for texture reflectance loss.')
        self.parser.add_argument('--cache_folder', type=str, default='fitting_cache',
                                 help='path for the cache folder')
        self.parser.add_argument('--nframes_shape', type=int, default=16,
                                 help='number of frames used to estimate shape coefficient in video fitting')
        self.parser.add_argument('--res_folder', type=str, required=False,
                                 help='output path for the image')
        self.parser.add_argument('--intr_matrix', type=list, default=None,   ###zxc
                                 help='focal of virtual camera')
        self.parser.add_argument('--crop_bbox', type=list, default=[],  ###zxc
                                 help='crop_bbox of the input image')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        #
        # args = vars(self.opt)
        #
        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        return self.opt


class ImageFittingOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--img_path', type=str, required=False,
                                 help='path for the image')
        self.parser.add_argument('--gpu', type=int, default=0,
                                 help='gpu device')


class VideoFittingOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--v_path', type=str, required=True,
                                 help='path for the video')
        self.parser.add_argument('--ngpus', type=int, default=1,
                                 help='gpu device')
        self.parser.add_argument('--nworkers', type=int, default=1,
                                 help='number of workers')
