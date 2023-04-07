import numpy as np
from termcolor import colored 

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import SmallUpdateBlock
from .extractor import ResidualBlock
from .corr import CoordSampler
from .resnet_encoder import resnet_multiimage_input
from .hr_encoder import hrnet18, hrnet32, hrnet48, hrnet64
from .hr_depth_decoder import HRDepthDecoder
from .utils.utils import PoseUpdate, Reprojections
from dualrefine.layers import *

from .lib.solvers import anderson, broyden
from .lib.grad import make_pair, backward_factory
from .utils.utils import autocast

import pdb


class DepthPose(nn.Module):
    """

    """

    def __init__(self, args):
        super(DepthPose, self).__init__()
        self.im_num = 0

        self.reproject_2 = Reprojections(args)

        if 'dropout' not in args:
            args.dropout = 0

        if 'alternate_corr' not in args:
            args.alternate_corr = False

        self.corr_fn = CoordSampler(args)
        
        odim = 64
        self.hidden_dim = hdim = 64
        self.context_dim = cdim = 64

        self.update_block = SmallUpdateBlock(args, input_dim=cdim, hidden_dim=hdim)

        # feature encoder
        self.hr_num_ch_enc = np.array([64, 18, 36, 72, 144])

        hrnets = {18: hrnet18,
                   32: hrnet32,
                   48: hrnet48,
                   64: hrnet64}
        # depth
        self._init_depth_net(args, hrnets)

        # feature encoder
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # pose encoder
        self._init_pose_net(args, 18)

        # Added the following for the DEQ models
        if not args.disable_wnorm:
            self.update_block._wnorm()

        self.f_solver = eval(args.f_solver)
        self.f_thres = args.f_thres
        self.eval_f_thres = int(self.f_thres * args.eval_factor)
        self.stop_mode = args.stop_mode

        self.hook = None
        self.args = args

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def freeze_backbone(self):
        for params in self.parameters():
            params.requires_grad = False
        refinement_modules = ['conv2', 'context', 'hidden', 'update_block']
        for refinement_module in refinement_modules:
            for params in eval(f'self.{refinement_module}').parameters():
                params.requires_grad = True
    
    def freeze_student(self):
        refinement_modules = ['conv2', 'context', 'hidden', 'update_block']
        for refinement_module in refinement_modules:
            for params in eval(f'self.{refinement_module}').parameters():
                params.requires_grad = False
        self.f_thres = 0
    
    def unfreeze_student(self):
        refinement_modules = ['conv2', 'context', 'hidden', 'update_block']
        for refinement_module in refinement_modules:
            for params in eval(f'self.{refinement_module}').parameters():
                params.requires_grad = True
        self.f_thres = self.args.f_thres

    def update_depth_bins(self, max_depth_bin, min_depth_bin, mean_depth_bin, median_depth_bin):
        self.max_depth_bin = max_depth_bin
        self.min_depth_bin = min_depth_bin
        self.mean_depth_bin = mean_depth_bin
        self.median_depth_bin = median_depth_bin
        self.reproject_2.update_depth_bins(max_depth_bin, min_depth_bin, mean_depth_bin, median_depth_bin)
        
    def _init_depth_net(self, args, hrnets):

        if args.num_layers not in hrnets:
            raise ValueError("{} is not a valid number of hrnet layers".format(args.num_layers))

        cnet_encoder = hrnets[args.num_layers](args.weights_init=="pretrained")
        self.cnet_stage1_cfg = cnet_encoder.stage1_cfg
        self.cnet_stage2_cfg = cnet_encoder.stage2_cfg
        self.cnet_stage3_cfg = cnet_encoder.stage3_cfg
        self.cnet_stage4_cfg = cnet_encoder.stage4_cfg
        
        self.cnet_layer0a = nn.Sequential(cnet_encoder.conv1,  cnet_encoder.bn1, cnet_encoder.relu)
        self.cnet_layer0b = nn.Sequential(cnet_encoder.conv2,  cnet_encoder.bn2, cnet_encoder.relu)
        self.cnet_layer1 = cnet_encoder.layer1
        self.cnet_transition1 = cnet_encoder.transition1
        self.cnet_stage2 = cnet_encoder.stage2
        self.cnet_transition2 = cnet_encoder.transition2
        self.cnet_stage3 = cnet_encoder.stage3
        self.cnet_transition3 = cnet_encoder.transition3
        self.cnet_stage4 = cnet_encoder.stage4

        # decoder
        self.decoder = HRDepthDecoder(self.hr_num_ch_enc)
        self.sigmoid = nn.Sigmoid()

        self.x2_chan = 64
        fmap_ch = 64
        self.net_chan = self.decoder.num_ch_dec[2]
        
        # matching feature
        self.conv2 = nn.Sequential(
                ResidualBlock(self.x2_chan, self.x2_chan, 'instance', stride=1),
                nn.Conv2d(self.x2_chan, fmap_ch, 3, padding=1))
        # context feature
        self.context = nn.Sequential(
                ResidualBlock(self.net_chan, self.net_chan, 'instance', stride=1),
                nn.Conv2d(self.net_chan, self.context_dim, 3, padding=1))
        # hidden feature
        self.hidden = nn.Sequential(
                ResidualBlock(self.net_chan, self.net_chan, 'instance', stride=1),
                nn.Conv2d(self.net_chan, self.hidden_dim, 3, padding=1))

    def _init_pose_net(self, args, num_layers):
        self.num_frames_to_predict_for = 1
        self.pnets = nn.ModuleList()
        for _ in range(3):
            pnets = nn.ModuleList()
            pnet = resnet_multiimage_input(num_layers, args.weights_init=="pretrained", 2)
            pnets.append(pnet)

            pnet_convs = nn.ModuleDict()
            in_ch = 2048 if num_layers > 34 else 512
            pnet_convs[("squeeze")] = nn.Conv2d(in_ch, 256, 1)
            pnet_convs[("pose0")] = nn.Conv2d(256, 256, 3, 1, 1)
            pnet_convs[("pose1")] = nn.Conv2d(256, 256, 3, 1, 1)
            pnet_convs[("pose2")] = nn.Conv2d(256, 6, 1)
            pnets.append(pnet_convs)

            relu = nn.ReLU()
            pnets.append(relu)

            self.pnets.append(pnets)

        if not args.disable_pose_updates:
            self.pose_update = PoseUpdate(args, self.x2_chan)

    def _make_hr_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.hr_in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.hr_in_planes = dim
        return nn.Sequential(*layers)
    
    def _log_convergence(self, result, name="FORWARD", color="yellow"):
        stop_mode = self.stop_mode
        alt_mode = "rel" if stop_mode == "abs" else "abs"
        diff_trace, alt_diff_trace = result[f'{stop_mode}_trace'], result[f'{alt_mode}_trace']
        stop_diff, alt_diff = min(diff_trace), min(alt_diff_trace)
        print(colored(f"{'TRAIN' if self.training else 'VALID'} | {name} | max depth : {self.max_depth_bin} min depth : {self.min_depth_bin} {stop_mode}_diff: {stop_diff}; {alt_mode}_diff: {alt_diff}; nstep: {result['nstep']}", f"{color}"))

    
    def _depth_net(self, image1, image2):
        B, C, H, W = image1.shape
        with autocast(enabled=self.args.mixed_precision):
            self.depth_feats = []

            features = []
            list18 = []
            list36 = []
            list72 = []
            
            x0 = torch.cat([image1, image2], 0)

            x0a = self.cnet_layer0a(x0)
            features.append(x0a[:B])
            x0b = self.cnet_layer0b(x0a)

            list18.append(x0b[:B])
            x1 = self.cnet_layer1(x0b[:B])

            x2 = x0b.clone()
            fmap1, fmap2 = self.conv2(x2).split(dim=0, split_size=B)
            
            x_list = []
            for i in range(self.cnet_stage2_cfg['NUM_BRANCHES']):
                if self.cnet_transition1[i] is not None:
                    x_list.append(self.cnet_transition1[i](x1))
                else:
                    x_list.append(x1)
            y_list = self.cnet_stage2(x_list)
            list18.append(y_list[0])
            list36.append(y_list[1])
            
            x_list = []
            for i in range(self.cnet_stage3_cfg['NUM_BRANCHES']):
                if self.cnet_transition2[i] is not None:
                    if i < self.cnet_stage2_cfg['NUM_BRANCHES']:
                        x_list.append(self.cnet_transition2[i](y_list[i]))
                    else:
                        x_list.append(self.cnet_transition2[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.cnet_stage3(x_list)
            list18.append(y_list[0])
            list36.append(y_list[1])
            list72.append(y_list[2])
            
            x_list = []
            for i in range(self.cnet_stage4_cfg['NUM_BRANCHES']):
                if self.cnet_transition3[i] is not None:
                    if i < self.cnet_stage3_cfg['NUM_BRANCHES']:
                        x_list.append(self.cnet_transition3[i](y_list[i]))
                    else:
                        x_list.append(self.cnet_transition3[i](y_list[-1]))
                        # here generate new scale features (downsample) 
                else:
                    x_list.append(y_list[i])
            x = self.cnet_stage4(x_list)
            list18.append(x[0])
            list36.append(x[1])
            list72.append(x[2])
            mixed_features = [list18] + [list36] + [list72] + [x[3]]

            decoder_features = features + mixed_features
            
            self.out_dec = {}
            # decoder
            feature144 = decoder_features[4]
            feature72 = decoder_features[3]
            feature36 = decoder_features[2]
            feature18 = decoder_features[1]
            feature64 = decoder_features[0]
            x72 = self.decoder.convs["72"](feature144, feature72)
            x36 = self.decoder.convs["36"](x72, feature36)
            x18 = self.decoder.convs["18"].no_relu_forward(x36, feature18)
            relu_x18 = torch.relu(x18)
            x9 = self.decoder.convs["9"](relu_x18, [feature64])
            x6 = self.decoder.convs["up_x9_1"](upsample(self.decoder.convs["up_x9_0"](x9)))
            
            self.out_dec[("disp", 0, 0)] = self.sigmoid(self.decoder.convs["dispConvScale0"](x6[:B]))
            self.out_dec[("disp", 1, 0)] = self.sigmoid(self.decoder.convs["dispConvScale1"](x9[:B]))
            self.out_dec[("disp", 2, 0)] = self.sigmoid(self.decoder.convs["dispConvScale2"](relu_x18[:B]))
            self.out_dec[("disp", 3, 0)] = self.sigmoid(self.decoder.convs["dispConvScale3"](x36[:B]))

            self.feature = [feature64]
            f18 = x18.clone()
            x2b = relu_x18.clone()

            inp = torch.relu(self.context(x2b))
            net = torch.tanh(self.hidden(x2b))
        
        return inp, net, f18, fmap1, fmap2, x2

    def _pose_net(self, pose_inp, invert, side):
        with autocast(enabled=self.args.mixed_precision):
            self.pose_feats = []
            x = self.pnets[side][0].conv1(torch.cat(pose_inp, 1))
            x = self.pnets[side][0].bn1(x)
            self.pose_feats.append(self.pnets[side][0].relu(x))
            try:
                self.pose_feats.append(self.pnets[side][0].layer1(self.pnets[side][0].maxpool(self.pose_feats[-1])))
            except:
                pdb.set_trace()
            self.pose_feats.append(self.pnets[side][0].layer2(self.pose_feats[-1]))
            self.pose_feats.append(self.pnets[side][0].layer3(self.pose_feats[-1]))
            self.pose_feats.append(self.pnets[side][0].layer4(self.pose_feats[-1]))

            last_features = [f[-1] for f in [self.pose_feats]]

            cat_features = [self.pnets[side][2](self.pnets[side][1]["squeeze"](f)) for f in last_features]
            cat_features = torch.cat(cat_features, 1)

            out = cat_features
            for i in range(3):
                out = self.pnets[side][1][f"pose{i}"](out)
                if i != 2:
                    out = self.pnets[side][2](out)

            out = out.mean(3).mean(2)

            out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

            axisangle = out[..., :3]
            translation = out[..., 3:]
            cam_T_cam = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=invert)
        
        return cam_T_cam

    def forward_pose(self, images, invert=False):
        sides = images[-1]
        images = images[:-1]
        image1, image2 = self._normalize_inputs(images)

        pose_inp = [image2, image1] if invert else [image1, image2]

        cam_T_cam = image1.new_zeros(
            (image1.shape[0], 4, 4),
            dtype=torch.float16 if self.args.mixed_precision else torch.float32)
        for i in range(3):
            if (sides==i).sum() > 0:
                cam_T_cam[sides==i] = self._pose_net([pose_inp[0][sides==i], pose_inp[1][sides==i]], invert, i)
                # cam_T_cam[sides==i] = self._pose_net([pose_inp[0][sides==i], pose_inp[1][sides==i]], invert, 0)
        return cam_T_cam

    def _normalize_inputs(self, images):
        image1 = (images[0] - .45) / 0.225
        image2 = (images[1] - .45) / 0.225

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        return image1, image2

    def _upsample_disp(self, disp, net):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        with autocast(enabled=self.args.mixed_precision):
            mask = .25 * self.update_block.mask(net)

        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)

        return up_disp.reshape(N, 1, 4*H, 4*W)
        
    def _decode(self, z_out, vec2list):
        for i, z_pred in enumerate(reversed(z_out)):
            net, depth = vec2list(z_pred)
            disp = self._depth_to_disp(depth)
            self.out_dec[("disp", 2, i+1)] = disp
            self.out_dec[("disp", 0, i+1)] = self._upsample_disp(disp, net)

    def _disp_to_depth(self, disp):
        return disp_to_depth(disp, self.args.min_depth, self.args.max_depth)[1]

    def _depth_to_disp(self, depth):
        min_disp = 1 / self.args.max_depth
        max_disp = 1 / self.args.min_depth
        scaled_disp = 1 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return disp

    def forward(self, images, inputs, invert=False, **kwargs):
        """ Estimate optical flow between pair of frames """

        seed = (images[0].get_device() == 0 and np.random.uniform(0,1) < 2e-3)
        self.K = inputs
        
        sides = images[-1]
        images = images[:-1]
        image1, image2 = self._normalize_inputs(images)
        self.image1, self.image2 = image1, image2

        # run the depth network
        # with autocast(enabled=self.args.mixed_precision):
        inp, net, f18, fmap1, fmap2, feat2 = self._depth_net(image1, image2)

        B, C, H, W = net.shape

        # extract corr volume
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        self.corr_fn.register(
            fmap1, fmap2, num_levels=self.args.num_levels)

        # init disp
        disp_2_0 = self.out_dec[("disp", 2, 0)]
        depth_2_0 = self._disp_to_depth(disp_2_0)
        self.out_dec[("disp", 0, 0)] = self._upsample_disp(disp_2_0, net)
        self.depth_init = self._disp_to_depth(self.out_dec[("disp", 0, 0)])

        # run the pose network
        pose_inp = [image2, image1] if invert else [image1, image2]
        # with autocast(enabled=self.args.mixed_precision):
        cam_T_cam = image1.new_zeros((B, 4, 4), dtype=torch.float16 if self.args.mixed_precision else torch.float32)
        for i in range(3):
            if (sides==i).sum() > 0:
                cam_T_cam[sides==i] = self._pose_net([pose_inp[0][sides==i], pose_inp[1][sides==i]], invert, i)
        self.poses_0 = cam_T_cam

        def list2vec(h, d):  # h is net, d is disps
            return torch.cat([h.view(B, -1), d.view(B, -1)], dim=1)

        def vec2list(hidden):
            h = hidden[:,:net.shape[1]*H*W].view_as(net)
            d = hidden[:,net.shape[1]*H*W:].view_as(disp_2_0)
            return h, d

        self.poses_hist, self.poses_updates_hist = [cam_T_cam], [cam_T_cam.new_ones((B))]
        def deq_func(hidden):
            h, depth = vec2list(hidden)

            depth = depth
            poses = self.poses

            # with autocast(enabled=self.args.mixed_precision):
            c, max_dx, ds = self.reproject_2.depth2epipolarcoords(poses, depth)
            corr = self.corr_fn(
                c, self.args.num_levels, self.args.num_cost_volume_head)

            with autocast(enabled=self.args.mixed_precision):
                new_h, delta = self.update_block(
                    h, inp, corr, depth, poses, None)
                
                new_depth = depth + torch.tanh(delta) * max_dx
                new_depth = torch.clamp(
                    new_depth, min=self.args.min_depth, max=self.args.max_depth)

                if not self.args.disable_evolving_pose_weight:
                    weight = self.update_block.weight(new_h)
                else:
                    weight = None

            if not self.args.disable_pose_updates:
                for i in range(self.args.num_pose_iter):
                    c_p, P2 = self.reproject_2.depth2gradcoords(poses, new_depth, inputs[("K", 2)])
                    poses, poses_update = self.pose_update.direct_align(
                        poses, inputs[("K", 2)], c_p, P2, weight)
                self.poses = poses
                self.poses_updates_hist.append(torch.norm(poses_update, dim=1)[:, 0])
                self.poses_hist.append(poses)                

            self.iter_num += 1

            if torch.isnan(new_h).sum() > 0 or torch.isinf(new_h).sum() > 0:
                pdb.set_trace()

            return list2vec(new_h, new_depth)

        self.update_block.reset()   # In case we use weight normalization, we need to recompute the weight with wg and wv
        z_star = list2vec(net, depth_2_0)
        self.iter_num = 0
        self.poses = self.poses_0.clone()

        self.reproject_2._reg_intrinsics(inputs[("K", 2)])

        if not self.args.disable_pose_updates:
            self.pose_update.compute_uncertainty(feat2)
            self.pose_update.compute_feat(fmap1, fmap2)
        
        return self._deq_forward(deq_func, vec2list, z_star,
                seed, **kwargs)


class DEQDepthPose(DepthPose):
    """

    """

    def __init__(self, args):

        super(DEQDepthPose, self).__init__(args)

        self.is_cuda = False
        
        if self.f_thres > 0:
            # Define gradient functions through the backward factory
            if args.n_losses > 1:
                n_losses = min(args.f_thres, args.n_losses)
                delta = int(args.f_thres // n_losses)
                self.indexing = [(k+1)*delta for k in range(n_losses)]
            else:
                self.indexing = [*args.indexing, args.f_thres]
            
            # By default, we use the same phantom grad for all corrections.
            # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
            indexing_pg = make_pair(self.indexing, args.phantom_grad)
            produce_grad = [
                    backward_factory(grad_type=pg, tau=args.tau, sup_all=args.sup_all) for pg in indexing_pg
                    ]
            if args.ift:
                # Enabling args.ift will replace the last gradient function by IFT.
                Warning("Not tested.")
                produce_grad[-1] = backward_factory(
                    grad_type='ift', safe_ift=args.safe_ift, b_solver=eval(args.b_solver),
                    b_solver_kwargs=dict(threshold=args.b_thres, stop_mode=args.stop_mode)
                    )

            self.produce_grad = produce_grad
    
    def _fixed_point_solve(self, deq_func, z_star, 
            seed=None, f_thres=None, **kwargs):
        if f_thres is None: f_thres = self.f_thres
        indexing = self.indexing if self.training else None

        with torch.no_grad():
            result = self.f_solver(deq_func, x0=z_star, threshold=f_thres, # To reuse previous coarse fixed points
                    eps=(0 if self.stop_mode == "abs" else 0), stop_mode=self.stop_mode, indexing=indexing)

            z_star, trajectory = result['result'], result['indexing']
        if seed and self.training:
            self._log_convergence(result, name="FORWARD", color="yellow")          
        
        return z_star, trajectory, min(result['rel_trace']), min(result['abs_trace'])

    def _deq_forward(self, deq_func, vec2list, z_star,
            seed=None, 
            **kwargs):
        # The code for DEQ version, where we use a wrapper. 
        if self.training:
            _, trajectory, rel_error, abs_error = self._fixed_point_solve(deq_func, z_star, seed=seed, *kwargs)
            
            # Set self.poses to the pose with the lowest update value
            if not self.args.disable_pose_updates:
                poses_updates_hist = torch.stack(self.poses_updates_hist)
                poses_hist = torch.stack(self.poses_hist)
                self.poses = torch.gather(poses_hist, 0, \
                    torch.argmin(poses_updates_hist, dim=0)[:, None, None].expand(poses_hist.shape[1:])[None])[0]

            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_out += produce_grad(self, z_pred, deq_func)  # See lib/grad.py for the backward pass implementations

            self._decode(z_out, vec2list)
            depth = vec2list(z_star)[1]

            if self.args.Pose_for_consistency_mask == 'Tstar':
                poses = self.poses.clone().detach()
            elif self.args.Pose_for_consistency_mask == 'T0':
                poses = self.poses_0.clone().detach()
            else:
                raise NotImplementedError
                
            num_depth_bins = self.reproject_2.num_depth_bins
            c0, ds0 = self.reproject_2.depthbins2coords(poses, depth)
            c0, ds0 = c0.detach(), ds0.detach()

            # self._display_epipolar(c0, uu=10, vv=12)

            # mask values landing outside the image (and near the border)
            # we want to ignore edge pixels of the lookup images and the current image
            # because of zero padding in ResNet
            # Masking of ref image border
            edge_mask = (c0[:, 0] >= 2.0) * (c0[:, 0] <= self.corr_fn.fmap1.shape[3] - 2) * \
                        (c0[:, 1] >= 2.0) * (c0[:, 1] <= self.corr_fn.fmap1.shape[2] - 2)
            edge_mask = edge_mask.float()
            # masking of current image
            current_mask = torch.zeros_like(edge_mask)
            current_mask[..., 2:-2, 2:-2] = 1.0
            edge_mask = (edge_mask * current_mask)[:, 0]
            # ?
            self.edge_mask = edge_mask.sum(1, keepdim=True) == num_depth_bins

            corr0 = self.corr_fn.__corr__(c0).detach()
            corrs = corr0.split(
                dim=1, split_size=num_depth_bins)
            ds0s = ds0[:, 0].split(
                dim=1, split_size=num_depth_bins)
            
            level_mask = 0
            max_d = torch.gather(
                ds0s[level_mask], 1, corrs[level_mask].min(1, keepdim=True)[1])

            mask = ((max_d - depth) / depth) < 1.0
            mask *= ((depth - max_d) / max_d) < 1.0
            self.mask = mask.detach()
            self.max_d = self._depth_to_disp(max_d.clone().detach())

            return self.out_dec, [self.poses_0, self.poses]

        else:
            self.im_num += 1
            # During inference, we directly solve for fixed point
            if self.eval_f_thres > 0:
                z_star, _, rel_error, abs_error = self._fixed_point_solve(
                    deq_func, z_star, f_thres=self.eval_f_thres, seed=seed)

                if not self.args.disable_pose_updates:
                    poses_updates_hist = torch.stack(self.poses_updates_hist)
                    poses_hist = torch.stack(self.poses_hist)
                    self.poses = torch.gather(poses_hist, 0, \
                        torch.argmin(poses_updates_hist, dim=0)[:, None, None].expand(poses_hist.shape[1:])[None])[0]
                    
                self._decode([z_star], vec2list)
                if self.args.combine_via_mask:
                    depth = vec2list(z_star)[1]

                    if self.args.Pose_for_consistency_mask == 'Tstar':
                        poses = self.poses.clone().detach()
                    elif self.args.Pose_for_consistency_mask == 'T0':
                        poses = self.poses_0.clone().detach()
                    else:
                        raise NotImplementedError
                        
                    num_depth_bins = self.reproject_2.num_depth_bins
                    c0, ds0 = self.reproject_2.depthbins2coords(poses, depth)
                    c0, ds0 = c0.detach(), ds0.detach()

                    # self._display_epipolar(c0, uu=10, vv=12)

                    # mask values landing outside the image (and near the border)
                    # we want to ignore edge pixels of the lookup images and the current image
                    # because of zero padding in ResNet
                    # Masking of ref image border
                    edge_mask = (c0[:, 0] >= 2.0) * (c0[:, 0] <= self.corr_fn.fmap1.shape[3] - 2) * \
                                (c0[:, 1] >= 2.0) * (c0[:, 1] <= self.corr_fn.fmap1.shape[2] - 2)
                    edge_mask = edge_mask.float()
                    # masking of current image
                    current_mask = torch.zeros_like(edge_mask)
                    current_mask[..., 2:-2, 2:-2] = 1.0
                    edge_mask = (edge_mask * current_mask)[:, 0]
                    # ?
                    self.edge_mask = edge_mask.sum(1, keepdim=True) == num_depth_bins

                    corr0 = self.corr_fn.__corr__(c0).detach()
                    corrs = corr0.split(
                        dim=1, split_size=num_depth_bins)
                    ds0s = ds0[:, 0].split(
                        dim=1, split_size=num_depth_bins)
                    
                    level_mask = 0
                    max_d = torch.gather(
                        ds0s[level_mask], 1, corrs[level_mask].min(1, keepdim=True)[1])

                    mask = ((max_d - depth) / depth) < 1.0
                    mask *= ((depth - max_d) / max_d) < 1.0
                    self.mask = mask.detach()
                    self.max_d = self._depth_to_disp(max_d.clone().detach())

                    self.out_dec[("disp", 2, 1)] = \
                        self.out_dec[("disp", 2, 1)] * self.mask + \
                        self.out_dec[("disp", 2, 0)] * (~self.mask)
                    mask_0 = self.mask.clone().detach()
                    mask_0 = F.interpolate(
                        mask_0.type(torch.float32),
                        [self.out_dec[("disp", 0, 1)].shape[2], self.out_dec[("disp", 0, 1)].shape[3]],
                        mode="nearest").type(torch.bool)
                    self.out_dec[("disp", 0, 1)] = \
                        self.out_dec[("disp", 0, 1)] * mask_0 + \
                        self.out_dec[("disp", 0, 0)] * (~mask_0)
            
            else:
                self.out_dec[("disp", 2, 1)] = self.out_dec[("disp", 2, 0)]
                self.out_dec[("disp", 0, 1)] = self.out_dec[("disp", 0, 0)]

            return self.out_dec, [self.poses_0, self.poses]

    def cuda(self):
        super().cuda()
        self.is_cuda = True

    def cpu(self):
        super().cpu()
        self.is_cuda = False

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
        