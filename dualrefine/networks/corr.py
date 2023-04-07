import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordSampler(nn.Module):
    def __init__(self, args):
        super(CoordSampler, self).__init__()
        self.args = args

    def register(self, fmap1, fmap2, num_levels=4):
        self.num_levels = num_levels
        self.f2_pyramid = []
        
        self.fmap1 = fmap1.clone()[..., None]
        self.fmap2 = fmap2.clone()[..., None]
        self.f2_pyramid.append(fmap2.clone())

        f2 = fmap2
        for i in range(self.num_levels-1):
            f2 = F.avg_pool2d(f2, 2, stride=2)
            self.f2_pyramid.append(f2)

    def __call__(self, coords, num_levels=1, num_head=1):
        batch, _, n1, d1, h1, w1 = coords.shape
        coords = coords.permute(2, 0, 4, 5, 3, 1).reshape(
            num_levels, batch, h1*w1, d1, 2)

        out_pyramid = []
        for i in range(num_levels):
            f2 = self.f2_pyramid[i]
            coord = coords[i]

            xgrid, ygrid = coord.split([1,1], dim=-1)
            xgrid = 2*(xgrid+0.5)/(w1) - 1
            ygrid = 2*(ygrid+0.5)/(h1) - 1

            grid = torch.cat([xgrid, ygrid], dim=-1)

            f2 = F.grid_sample(f2, grid, align_corners=False)
            f2 = f2.view(batch, -1, h1, w1, d1)

            corr = torch.abs(self.fmap1 - f2)
            corr = corr.view(batch, num_head, -1, h1, w1, d1).mean(2)
            corr = corr.permute(0, 2, 3, 1, 4).reshape(batch, h1, w1, -1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def __corr__(self, coords, num_levels=1, num_head=1):
        batch, _, n1, d1, h1, w1 = coords.shape
        coords = coords.permute(2, 0, 4, 5, 3, 1).reshape(
            num_levels, batch, h1*w1, d1, 2)

        out_pyramid = []
        for i in range(num_levels):
            f2 = self.f2_pyramid[i]
            coord = coords[i]

            xgrid, ygrid = coord.split([1,1], dim=-1)
            xgrid = 2*(xgrid+0.5)/(w1) - 1
            ygrid = 2*(ygrid+0.5)/(h1) - 1

            grid = torch.cat([xgrid, ygrid], dim=-1)

            f2 = F.grid_sample(f2, grid, align_corners=False)
            f2 = f2.view(batch, -1, h1, w1, d1)

            corr = torch.abs(self.fmap1 - f2).mean(1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def _update_fmap1(self, fmap1):
        self.fmap1 = fmap1.clone()[..., None]
