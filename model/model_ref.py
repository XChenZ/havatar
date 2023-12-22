class ConditionalTriplaneNeRFModel_multiRender_split_view(torch.nn.Module):
    def __init__(self, XYZ_bounding, num_encoding_fn_xyz=8, latent_code_dim=32, triPlane_feat_dim=32, rgb_feat_dim=32, triplane_res=256, use_emb=True,
                 enc_mode='split', sh_deg=2, cond_latent=True, cond_c_dim=0):
        super(ConditionalTriplaneNeRFModel_multiRender_split_view, self).__init__()
        self.name = 'ConditionalTriplaneNeRFModel_multiRender_split_view'

        self.pos_embedder, self.dim_xyz = get_embedder(multires=num_encoding_fn_xyz, input_dims=3, include_input=False) # 搭配ipe
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
                                       zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=7+13)
        elif self.two_head:
            self.XY_gen = StyleGAN_zxc_twoHead(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=8,
                                               split_size=128,
                                               zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=[7, 13])
        else:
            self.XY_gen = StyleGAN_zxc(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=16,
                                       zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=7)
            self.YZ_gen = StyleGAN_zxc(out_ch=triPlane_feat_dim, out_size=triplane_res, style_dim=latent_code_dim, middle_size=16,
                                       zero_latent=False, zero_noise=True, no_skip=True, n_mlp=4, inp_size=256, inp_ch=13)

        print('XYZ_bounding', XYZ_bounding)
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
            # pose = canonical_condition['pose']
            # pose = pose.reshape(pose.shape[0], -1)
            if self.cond_latent:
                inp_latents = [torch.cat([latents, cond_c], -1)] if self.cond_c_dim > 0 else [latents]
            else:
                inp_latents = [self.zero_latent.expand(latents.shape[0], -1)]
        else:
            inp_latents = None

        front_render_cond = canonical_condition['front_render_cond']    # [B, 7, H, W]
        left_render_cond = canonical_condition['left_render_cond'].flip(dims=[3])  # [B, 7, H, W]   #右平面满足左上角(-1,-1)，右下角(1, 1)
        right_render_cond = canonical_condition['right_render_cond']  # [B, 7, H, W]
        if self.shared_backbone:
            conditonplane_embedding, _ = self.XY_gen(inp_latents, torch.cat([front_render_cond, left_render_cond[:, :-1], right_render_cond], dim=1))
            conditonXYplane_embedding, conditonYZplane_embedding = conditonplane_embedding[:, :self.triPlane_feat_dim], \
                                                                   conditonplane_embedding[:, self.triPlane_feat_dim:]
        elif self.two_head:
            conditonXYplane_embedding, conditonYZplane_embedding = self.XY_gen(
                inp_latents, [front_render_cond, torch.cat([left_render_cond[:, :6], right_render_cond], dim=1)])
        else:
            conditonXYplane_embedding, _ = self.XY_gen(inp_latents, front_render_cond)
            conditonYZplane_embedding, _ = self.YZ_gen(inp_latents, torch.cat([left_render_cond[:, :-1], right_render_cond], dim=1))

        # conditionPlanes_feat = self.Plane_embeddings.unsqueeze(1).expand(-1, batch_num, -1, -1, -1) + \
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
        self.pts_triPlane_feat = pts_triPlane_feat.reshape(-1, pts_triPlane_feat.shape[-1] * pts_triPlane_feat.shape[-2]) # [BN, C * 3]
        # self.pts_mask = ((torch.abs(inp_pts) > 0.9).sum(dim=-1) == 0).float().reshape(self.pts_triPlane_feat.shape[0])  ## 这种分离的写法会多占显存，最好把该函数和forward融合

    def forward(self, inp, expr=None, latent_code=None, ipe_pts=None, return_lastFeat=False):
        xyz, dirs = inp[..., :3], inp[..., 3:]
        pts_feat = self.pts_triPlane_feat
        if self.use_emb:
            xyz_emb = ipe_pts if ipe_pts is not None else self.pos_embedder(xyz)
            pts_feat = torch.cat([pts_feat, xyz_emb], -1)
        if self.dim_latent_code > 0:
            pts_feat = torch.cat([pts_feat, latent_code], -1)
        x = pts_feat
        # out = torch.zeros(pts_feat.shape[0], 4, dtype=torch.float32, device=pts_feat.device)
        # out = torch.cat([-100 * torch.ones(pts_feat.shape[0], 3, dtype=torch.float32, device=pts_feat.device),
        #                 torch.zeros(pts_feat.shape[0], 1, dtype=torch.float32, device=pts_feat.device)], -1)
        # x = pts_feat[self.pts_mask > 0]
        # print(x.shape[0] / pts_feat.shape[0])
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
        if return_lastFeat:
            rgb = torch.cat([rgb, x], dim=-1)
        return torch.cat((rgb, alpha), dim=-1)
        # out[self.pts_mask>0] = torch.cat((rgb, alpha), dim=-1)
        # return out

class StyleGAN_zxc_twoHead(nn.Module):
    def __init__(self, out_ch, out_size, style_dim, mlp_dim=32, n_mlp=0, middle_size=8, split_size=64, zero_latent=False, zero_noise=False, no_skip=False,
                 channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, inp_size=0, inp_ch=[]):
        super().__init__()
        self.no_skip = no_skip
        # self.out_rgb = out_rgb
        # self.inp_size = inp_size
        self.style_dim = mlp_dim
        self.middle_log_size = int(math.log(middle_size, 2))
        self.split_log_size = int(math.log(split_size, 2))
        self.n_mlp = n_mlp
        if n_mlp > 0:
            layers = [PixelNorm(),
                      EqualLinear(style_dim, mlp_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
            for i in range(n_mlp-1):
                layers.append(
                    EqualLinear(
                        mlp_dim, mlp_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                    )
                )

            self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # in_channel = self.channels[self.inp_size]  # 128
        # add new layer here
        # self.dwt = HaarTransform(3)

        self.log_size = int(math.log(out_size, 2))
        print('StyleGAN conditon img!')

        in_channel = self.channels[inp_size // 2]  # 128
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()
        self.comb_convs.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in = ConvLayer(inp_ch[0], in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.split_log_size - 1, -1):  #
            out_channel = self.channels[2 ** i]  # (inp_size/2)->->(8*512)
            # self.from_rgbs.append(FromRGB(in_channel, inp_ch[0], downsample=True, use_wt=False))  # //2
            self.from_rgbs.append(None)  # //2
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel))  # //2
            self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            in_channel = out_channel

        in_channel = self.channels[inp_size // 2]  # 128
        self.from_rgbs1 = nn.ModuleList()
        self.cond_convs1 = nn.ModuleList()
        self.comb_convs1 = nn.ModuleList()
        self.comb_convs1.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in1 = ConvLayer(inp_ch[1], in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.split_log_size - 1, -1):  #
            out_channel = self.channels[2 ** i]  # (inp_size/2)->->(8*512)
            self.from_rgbs1.append(FromRGB(in_channel, inp_ch[1], downsample=True, use_wt=False))  # //2
            # self.from_rgbs1.append(None)  # //2
            self.cond_convs1.append(ConvBlock(in_channel, out_channel, blur_kernel))  # //2
            self.comb_convs1.append(ConvLayer(out_channel * 2, out_channel, 3))
            in_channel = out_channel

        ################################ shared bakcbone
        self.convs = nn.ModuleList()  # [8, 512]->[16, 512]->[32, 512]
        self.to_rgbs = nn.ModuleList()

        self.input = ConstantInput(self.channels[middle_size], size=middle_size)
        self.conv1 = StyledConv(
            self.channels[middle_size], self.channels[middle_size], 3, self.style_dim, blur_kernel=blur_kernel
        )
        if self.no_skip:
            self.conv_out = ConvLayer(self.channels[out_size], out_ch, 1)
            self.conv_out1 = ConvLayer(self.channels[out_size], out_ch, 1)
        else:
            self.to_rgb1 = ToRGB(self.channels[middle_size], self.style_dim, out_channel=out_ch * 4, upsample=False, use_wt=False)

        in_channel = self.channels[middle_size]
        for i in range(self.middle_log_size + 1, self.split_log_size + 1):  # 4, 5, 6, 7, 8, 9
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True, blur_kernel=blur_kernel,))
            self.convs.append(StyledConv(out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel))
            if not self.no_skip:
                self.to_rgbs.append(ToRGB(in_channel=out_channel, style_dim=self.style_dim, out_channel=out_ch * 4, use_wt=False))
            else:
                self.to_rgbs.append(None)

            in_channel = out_channel

        split_init_channel = in_channel
        ################################ head 1
        self.convs_head = nn.ModuleList()
        self.to_rgbs_head = nn.ModuleList()
        for i in range(self.split_log_size + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs_head.append(StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True, blur_kernel=blur_kernel,))
            self.convs_head.append(StyledConv(out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel))
            if not self.no_skip:
                self.to_rgbs_head.append(ToRGB(in_channel=out_channel, style_dim=self.style_dim, out_channel=out_ch * 4, use_wt=False))
            else:
                self.to_rgbs_head.append(None)

            in_channel = out_channel

        ################################ head 2
        in_channel = split_init_channel
        self.convs_head1 = nn.ModuleList()
        self.to_rgbs_head1 = nn.ModuleList()
        for i in range(self.split_log_size + 1, self.log_size + 1):  # 4, 5, 6, 7, 8, 9
            out_channel = self.channels[2 ** i]
            self.convs_head1.append(StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True, blur_kernel=blur_kernel,))
            self.convs_head1.append(StyledConv(out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel))
            if not self.no_skip:
                self.to_rgbs_head1.append(ToRGB(in_channel=out_channel, style_dim=self.style_dim, out_channel=out_ch * 4, use_wt=False))
            else:
                self.to_rgbs_head1.append(None)
            in_channel = out_channel


        self.noises = nn.Module()
        self.num_layers = (self.log_size - self.middle_log_size) * 2 + 1
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 8) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        self.n_latents = [self.split_log_size * 2 - (self.middle_log_size * 2) + 1,
                          self.log_size * 2 - (self.split_log_size * 2),
                          self.log_size * 2 - (self.split_log_size * 2)]
        self.n_latent = sum(self.n_latents)

        if zero_noise:
            self.zero_noise = self.make_noise(device='cuda:0', zero_noise=True)
        else:
            self.zero_noise = None

        if zero_latent:
            self.register_buffer('zero_latents', torch.zeros(1, self.n_latent, self.style_dim))
        else:
            self.zero_latents = None

    def make_noise(self, device, zero_noise=False):
        func = torch.zeros if zero_noise else torch.randn
        noises = [torch.randn(1, 1, 2 ** self.middle_log_size, 2 ** self.middle_log_size, device=device)]
        for i in range(self.middle_log_size + 1, self.split_log_size + 1):
            for _ in range(2):
                noises.append(func(1, 1, 2 ** i, 2 ** i, device=device))
        for i in range(self.split_log_size + 1, self.log_size + 1):
            for _ in range(2):
                noises.append(func(1, 1, 2 ** i, 2 ** i, device=device))
        for i in range(self.split_log_size + 1, self.log_size + 1):
            for _ in range(2):
                noises.append(func(1, 1, 2 ** i, 2 ** i, device=device))
        # if zero_noise:
        #     for i in range(len(noises)):
        #         if i < len(noises) - 2:
        #             noises[i] = None
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, styles, n_latent=None, inject_index=None):
        styles = [self.style(s) for s in styles]
        n_latent_ = self.n_latent if n_latent is None else n_latent
        if len(styles) < 2:
            inject_index = n_latent_
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
        else:
            if inject_index is None:
                inject_index = random.randint(1, n_latent_ - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, n_latent_ - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)

        return latent

    def forward(
            self,
            styles,
            cond_imgs,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        batch_num = cond_imgs[0].shape[0]
        if self.zero_latents is None:
            if not input_is_latent:
                assert self.n_mlp > 0
                styles = [self.style(s) for s in styles]

            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            latent = self.zero_latents.expand(batch_num, -1, -1)

        if self.zero_noise is None:
            if noise is None:
                if randomize_noise:
                    noise = [None] * self.n_latent  ## zxc
                else:
                    noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        else:
            noise = self.zero_noise

        all_latent, all_noise = latent, noise
        # print(all_latent.shape, len(all_noise), self.n_latents)
        ########################### shared backbone
        latent, noise = all_latent[:, :self.n_latents[0]], all_noise[:self.n_latents[0]]
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = None
        if not self.no_skip:
            skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            if not self.no_skip:
                skip = to_rgb(out, latent[:, i + 2], skip)
            # print('backbone', out.shape)
            i += 2

        split_out, split_skip = out, skip

        ########################### head 1
        latent, noise = all_latent[:, self.n_latents[0]:sum(self.n_latents[:2])], all_noise[self.n_latents[0]:sum(self.n_latents[:2])]
        cond_img = cond_imgs[0]
        cond_out = self.conv_in(cond_img)  ### None
        cond_list = [cond_out]  ### []
        cond_num = 0
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            # cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            # print('Down', cond_img.shape, cond_out.shape)
            cond_list.append(cond_out)
            cond_num += 1

        i = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs_head[::2], self.convs_head[1::2], noise[::2], noise[1::2], self.to_rgbs_head
        ):
            # print('head1', out.shape, cond_list[- (i // 2 + 1)].shape, latent.shape)
            out = torch.cat([out, cond_list[- (i // 2 + 1)]], dim=1)
            out = self.comb_convs[- (i // 2 + 1)](out)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            if not self.no_skip:
                skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        if not self.no_skip:
            image = skip
        else:
            image = self.conv_out(out)

        ########################### head 2
        latent, noise = all_latent[:, sum(self.n_latents[:2]):sum(self.n_latents)], all_noise[sum(self.n_latents[:2]):sum(self.n_latents)]
        out, skip = split_out, split_skip
        cond_img = cond_imgs[1]
        cond_out = self.conv_in1(cond_img)
        cond_list = [cond_out]
        cond_num = 0
        for from_rgb, cond_conv in zip(self.from_rgbs1, self.cond_convs1):
            # cond_img, cond_out = from_rgb(cond_img, cond_out)   #############之前都忘删了。。。。。
            cond_out = cond_conv(cond_out)
            # print('Down', cond_img.shape, cond_out.shape)
            cond_list.append(cond_out)
            cond_num += 1

        i = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs_head1[::2], self.convs_head1[1::2], noise[::2], noise[1::2], self.to_rgbs_head1
        ):
            # print('head2', out.shape, cond_list[- (i // 2 + 1)].shape, latent.shape)
            out = torch.cat([out, cond_list[- (i // 2 + 1)]], dim=1)
            out = self.comb_convs1[- (i // 2 + 1)](out)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)

            if not self.no_skip:
                skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        if not self.no_skip:
            image1 = skip
        else:
            image1 = self.conv_out1(out)

        return image, image1