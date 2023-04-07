import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from typing import Tuple

from .losses import scaled_barron
from ..extractor import ResidualBlock
from dualrefine.layers import se3_exp

import pdb


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*(xgrid+0.5)/(W) - 1
    ygrid = 2*(ygrid+0.5)/(H) - 1
    # print(xgrid.max(), xgrid.min(), ygrid.max(), ygrid.min())

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=False)

    # Enable higher order grad for JR
    # img = grid_sample(img, grid)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=False)



class Reprojections(nn.Module):
    def __init__(self, args):
        super(Reprojections, self).__init__()
        self.Y, self.X = None, None
        self.r = args.corr_radius
        self.delta = nn.parameter.Parameter(torch.tensor([1.]))
        if not args.disable_pose_updates:
            self.delta_p = nn.parameter.Parameter(torch.tensor([1.]))

        self.num_depth_bins = 96
        self.args = args

    def update_depth_bins(self, max_depth_bin, min_depth_bin, mean_depth_bin, median_depth_bin):
        self.max_depth_bin = max_depth_bin
        self.min_depth_bin = min_depth_bin
        self.mean_depth_bin = mean_depth_bin
        self.median_depth_bin = median_depth_bin

    def _iproj(self, Z):
        """ pinhole camera inverse projection """
        ht, wd = Z.shape[-2:]
        if self.X is None:
            recompute = True
        else:
            if self.X.shape != Z.shape:
                recompute = True
            else:
                recompute = False
        
        if recompute:
            self.y, self.x = torch.meshgrid(
                torch.arange(ht).to(Z.device).float(),
                torch.arange(wd).to(Z.device).float())

            self.X = (self.x[None] - self.cx[:, None, None]) / self.fx[:, None, None]
            self.Y = (self.y[None] - self.cy[:, None, None]) / self.fy[:, None, None]

            self.X, self.Y = self.X[:, None, None], self.Y[:, None, None]
            
        pts = torch.cat([Z*self.X, Z*self.Y, Z, torch.ones_like(Z)], dim=1)

        return pts

    def _proj(self, Xs):
        """ pinhole camera projection """
        X, Y, Z, D = Xs.unbind(dim=1)

        # Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
        d = torch.clamp(1.0 / Z, max=100)

        x = self.fx[:, None] * (X * d) + self.cx[:, None]
        y = self.fy[:, None] * (Y * d) + self.cy[:, None]

        coords = torch.stack([x, y], dim=1) # (B, 2, -1)

        return coords

    def _reg_intrinsics(self, intrinsics):

        self.fx, self.fy, self.cx, self.cy = \
            intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    def minmax_gap(self, r):
        return (self.max_depth_bin - self.min_depth_bin) * 4 * r / self.num_depth_bins

    def depth2epipolarcoords(self, poses, depths):
        bsz, _, ht, wd = depths.shape

        r = self.r

        dx = torch.linspace(-r, r, 2*r+1, device=depths.device)[None, None, :, None, None]
        depths = depths[:, None]

        deltas = []
        if self.args.gap_factor == 'depth':
            gap_factor = depths
        else:
            gap_factor = eval('self.' + self.args.gap_factor)(r)
        dd = F.softplus(self.delta)
        
        gap = dd * gap_factor / self.args.gap_factor_depth_ratio / r
        for level in range(self.args.num_levels):
            delta = (2**level) * depths.new_ones(depths.shape) * gap  # / r

            if level == 0:
                max_dx = (dx*delta).max(dim=2, keepdim=True)[0][:, 0]

            deltas.append(dx * delta)
        deltas = torch.cat(deltas, 2)
            
        depths = depths + deltas

        X0 = self._iproj(depths)
        X1 = poses.type(X0.dtype) @ X0.reshape(bsz, 4, -1)
        c1 = self._proj(X1).reshape(bsz, 2, self.args.num_levels, 2*r+1, ht, wd)
        
        if c1.isinf().sum() > 0 or c1.isnan().sum() > 0:
            pdb.set_trace()

        return c1, max_dx, depths

    def depth2gradcoords(self, poses, depths, intrinsics):
        bsz, _, ht, wd = depths.shape

        X0 = self._iproj(depths[:, None])
        X1 = poses.type(X0.dtype) @ X0.reshape(bsz, 4, -1)
        c1 = self._proj(X1).reshape(bsz, 2, 1, 1, ht, wd)

        p_dx = torch.tensor([1., 0.], device=depths.device).reshape(1, 2, 1, 1, 1, 1)
        p_dy = torch.tensor([0., 1.], device=depths.device).reshape(1, 2, 1, 1, 1, 1)
        
        p_delta = torch.cat([p_dx, -p_dx, p_dy, -p_dy], 3)
        
        c1 = torch.cat([c1, c1 + p_delta], 3)
        
        if c1.isinf().sum() > 0 or c1.isnan().sum() > 0:
            pdb.set_trace()

        return c1, X1

    def depthbins2coords(self, poses, depths):
        bsz, _, _, ht, wd = self.X.shape
        
        if self.args.use_depth_bins_for_masking:
            depths = torch.linspace(
                self.min_depth_bin, self.max_depth_bin, self.num_depth_bins, device=poses.device)
            depths = depths[None, None, :, None, None].repeat(bsz, 1, 1, ht, wd)
        else:
            # lin = torch.linspace(0.1, 4., self.num_depth_bins, device=poses.device)
            # depths = lin[None, None, :, None, None] * (depths[:, None] - self.args.min_depth) + self.args.min_depth
            # depths = torch.clamp(depths, max=4.*self.args.max_depth)

            lin = torch.linspace(0, 1, self.num_depth_bins, device=poses.device)
            depths_ = 8 * (depths - self.args.min_depth) + self.args.min_depth
            depths_ = torch.clamp(depths_, max=self.args.max_depth)
            lin_ = (depths_ - self.args.min_depth) / (depths - self.args.min_depth)
            lin = lin[None, None, :, None, None] * lin_[:, None]
            depths = lin * (depths[:, None] - self.args.min_depth) + self.args.min_depth

        X0 = self._iproj(depths)
        X1 = poses.type(X0.dtype) @ X0.reshape(bsz, 4, -1)
        c1 = self._proj(X1).reshape(bsz, 2, 1, self.num_depth_bins, ht, wd)

        return c1, depths
    

class PoseUpdate(nn.Module):
    def __init__(self, args, inp_dim, norm_fn='batch'):
        super(PoseUpdate, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=inp_dim)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(inp_dim)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(inp_dim)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.in_planes = inp_dim
        self.weights = nn.Sequential(
            self._make_layer(inp_dim, stride=1),
            nn.Conv2d(inp_dim, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU())
        
        self.feats = nn.Sequential(
            self._make_layer(inp_dim, stride=1),
            nn.Conv2d(inp_dim, 16, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.loss_fn = scaled_barron(0, 0.1)
        self.args = args

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def compute_uncertainty(self, feats):
        if not self.args.disable_fixed_pose_weight:
            with autocast(enabled=self.args.mixed_precision):
                self.src_w, self.tgt_w = self.weights(feats).split(dim=0, split_size=feats.shape[0]//2)
            self.src_w, self.tgt_w = 1/(1+self.src_w.float()), 1/(1+self.tgt_w.float())
        else:
            bsz, _, ht, wd = feats.shape
            self.src_w, self.tgt_w = feats[0].new_ones((bsz//2, 1, ht, wd)), feats[1].new_ones((bsz//2, 1, ht, wd))

    def compute_feat(self, fmap1, fmap2):
        self.src_feat, self.tgt_feat = fmap1.float(), fmap2.float()

    def direct_align(self, poses, calib_K, p2, P2, weight):
        src_feat, tgt_feat = self.src_feat.clone(), self.tgt_feat.clone()

        batch_size, channels, height, width = src_feat.shape

        warped_tgt_feat, warped_tgt_gradients = self.sample_tgt(tgt_feat, p2)
        
        X, Y, Z = P2[:, 0], P2[:, 1], P2[:, 2]

        fx, fy = calib_K[:, 0, 0].reshape(-1, 1), calib_K[:, 1, 1].reshape(-1, 1)
        fx_x_Z_inv, fy_x_Z_inv = fx / Z, fy / Z
        fx_x_X_x_Z2_inv, fy_x_Y_x_Z2_inv = fx_x_Z_inv * X / Z, fy_x_Z_inv * Y / Z
        zeros = torch.zeros(fx_x_X_x_Z2_inv.shape, device=src_feat.device)

        J_pixel_xi = torch.stack((
            torch.stack((
                fx_x_Z_inv,
                zeros,
                -fx_x_X_x_Z2_inv,
                -fx_x_X_x_Z2_inv * Y,
                fx + fx_x_X_x_Z2_inv * X,
                -fx_x_Z_inv * Y), 1),
            torch.stack((
                zeros,
                fy_x_Z_inv,
                -fy_x_Y_x_Z2_inv,
                -fy - fy_x_Y_x_Z2_inv * Y,
                fy_x_Y_x_Z2_inv * X,
                fy_x_Z_inv * X), 1)), 1)
                
        J_pixel_xi = J_pixel_xi.permute(0, 3, 1, 2)
        J_img_pixel = warped_tgt_gradients.reshape(batch_size, channels, height*width, 2)
        J_img_pixel = J_img_pixel.permute(0, 2, 1, 3)

        with torch.cuda.amp.autocast(enabled=False):
            J = -J_img_pixel @ J_pixel_xi

        # Simple L1 error for now
        res = (src_feat-warped_tgt_feat).permute(0, 2, 3, 1).reshape(-1, height*width, channels, 1)

        # # compute the cost and aggregate the weights
        if self.args.robust_pose_loss:
            cost = (res[..., 0]**2).sum(-1).reshape(batch_size, 1, height, width)
            cost, w_loss, _ = self.loss_fn(cost)
            valid = self._mask_in_image(p2[:, :, 0, 0], (width, height), pad=2)
            weights = w_loss * valid.float()
        
        w = (self.src_w * self.warped_tgt_w)
        if weight is not None:
            w *= weight
        if self.args.robust_pose_loss:
            w *= weights
        # self.w = w.clone()
        JW = J * w.reshape(batch_size, height*width, 1, 1)
        with torch.cuda.amp.autocast(enabled=False):
            Hessian = JW.transpose(2, 3) @ J
            
        H = (Hessian).sum(1)

        b = (-res * JW).sum(2).sum(1)

        try:
            L = torch.linalg.cholesky(H)
            if torch.isnan(L).sum() > 0:
                raise RuntimeError
        except:
            try:
                update = torch.linalg.solve(H, b[..., None])
            except:
                return poses, poses
        else:
            update = torch.cholesky_solve(b[..., None], L)

        new_poses = self._update_pose(poses, update)

        if torch.isnan(new_poses).sum() > 0:
            pdb.set_trace()

        return new_poses, update

    def sample_tgt(self, tgt_feat, p2):
        batch, _, n1, d1, h1, w1 = p2.shape
        p2 = p2.permute(2, 0, 4, 5, 3, 1).reshape(batch, h1*w1, d1, 2)

        xgrid, ygrid = p2.split([1,1], dim=-1)
        xgrid = 2*(xgrid+0.5)/(w1) - 1
        ygrid = 2*(ygrid+0.5)/(h1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)

        f = F.grid_sample(tgt_feat, grid, align_corners=False)
        f = f.view(batch, -1, h1, w1, d1)

        warped_tgt_feat = f[..., 0]
        warped_tgt_gradients = torch.stack([
            (f[..., 1] - f[..., 2])/2, (f[..., 3] - f[..., 4])/2], dim=-1)

        grid_0 = grid[:, :, :1]
        self.warped_tgt_w = F.grid_sample(self.tgt_w.type(grid_0.dtype), grid_0, align_corners=False).reshape(
            batch, 1, h1, w1)

        return warped_tgt_feat, warped_tgt_gradients

    def _update_pose(self, pose, update):
        return torch.bmm(se3_exp(update).type(pose.dtype), pose)

    def _mask_in_image(self, pts, image_size: Tuple[int, int], pad: int = 1):
        w, h = image_size
        image_size_ = torch.tensor([w-pad-1, h-pad-1]).to(pts).reshape(1, 2, 1, 1)
        return torch.all((pts >= pad) & (pts <= image_size_), 1, keepdim=True)
    