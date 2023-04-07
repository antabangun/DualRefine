import torch
import torch.nn as nn
import torch.nn.functional as F
from .lib.optimizations import weight_norm

import pdb

class ConvHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_dim=128, act=None):
        super(ConvHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        # self.gn1 = nn.GroupNorm(8, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()
    
    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(module=self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(module=self.conv2, names=['weight'], dim=0)

    def reset(self):
        for name in ['conv1', 'conv2']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, x):
        # return torch.tanh(self.conv2(self.relu(self.conv1(x))))
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.conv_net = nn.Conv2d(hidden_dim, 2*hidden_dim, 3, padding=1)
        self.conv_inp = nn.Conv2d(input_dim, 2*hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

        self.w = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)

        self.convz_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        self.convr_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        self.convq_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        
        self.h_planes = hidden_dim
        self.args = args

    def _wnorm(self):
        self.conv_net, self.conv_net_fn = weight_norm(module=self.conv_net, names=['weight'], dim=0)
        self.conv_inp, self.conv_inp_fn = weight_norm(module=self.conv_inp, names=['weight'], dim=0)
        self.w, self.w_fn = weight_norm(module=self.w, names=['weight'], dim=0)
        self.convz_glo, self.convz_glo_fn = weight_norm(module=self.convz_glo, names=['weight'], dim=0)
        self.convr_glo, self.convr_glo_fn = weight_norm(module=self.convr_glo, names=['weight'], dim=0)
        self.convq_glo, self.convq_glo_fn = weight_norm(module=self.convq_glo, names=['weight'], dim=0)
        self.convq, self.convq_fn = weight_norm(module=self.convq, names=['weight'], dim=0)

    def reset(self):
        names = ['conv_net', 'conv_inp', 'w', 'convz_glo', 'convr_glo', 'convq_glo', 'convq']
        for name in names:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, h, x):
        bsz, ch, ht, wd = h.shape
        glo = torch.sigmoid(self.w(h)) * h
        glo = glo.view(bsz, ch, ht*wd).mean(-1).view(bsz, ch, 1, 1)

        z_net, r_net = self.conv_net(h).split(
            [self.h_planes, self.h_planes], dim=1)
        z_inp, r_inp = self.conv_inp(x).split(
            [self.h_planes, self.h_planes], dim=1)

        z = torch.sigmoid(z_net + z_inp + self.convz_glo(glo))
        r = torch.sigmoid(r_net + r_inp + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + self.convq_glo(glo))

        h = (1-z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = (2*args.corr_radius + 1)*args.num_levels
        cor_planes *= args.num_cost_volume_head
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convc2 = nn.Conv2d(96, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.convp1 = nn.Linear(6, 128)
        self.convp2 = nn.Linear(128, 128)
        self.conv = nn.Conv2d(96, 63, 3, padding=1)

    def _wnorm(self):
        self.convc1, self.convc1_fn = weight_norm(module=self.convc1, names=['weight'], dim=0)
        self.convc2, self.convc2_fn = weight_norm(module=self.convc2, names=['weight'], dim=0)
        self.convf1, self.convf1_fn = weight_norm(module=self.convf1, names=['weight'], dim=0)
        self.convf2, self.convf2_fn = weight_norm(module=self.convf2, names=['weight'], dim=0)
        self.convp1, self.convp1_fn = weight_norm(module=self.convp1, names=['weight'], dim=0)
        self.convp2, self.convp2_fn = weight_norm(module=self.convp2, names=['weight'], dim=0)
        self.conv, self.conv_fn = weight_norm(module=self.conv, names=['weight'], dim=0)

    def reset(self):
        for name in ['convc1', 'convc2', 'convf1', 'convf2', 'convp1', 'convp2', 'conv']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, depths, poses, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dep = F.relu(self.convf1(depths))
        dep = F.relu(self.convf2(dep))
        # pos = F.relu(self.convp1(poses[:, 0]))
        # pos = F.relu(self.convp2(pos))
        cor_dep = torch.cat([cor, dep], dim=1) #+ pos[..., None, None]
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, depths], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, input_dim=64, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(args, hidden_dim=hidden_dim, input_dim=64+input_dim)
        self.conv_head = ConvHead(hidden_dim, hidden_dim=64, out_dim=1)
        
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16*9, 1, padding=0))
        if not args.disable_evolving_pose_weight:
            self.weight = nn.Sequential(
                nn.Conv2d(hidden_dim, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1, padding=0), nn.Sigmoid())

        self.args = args

    def _wnorm(self):
        print("Applying weight normalization to SmallUpdateBlock")
        self.encoder._wnorm()
        self.gru._wnorm()
        self.conv_head._wnorm()
    
    def reset(self):
        self.encoder.reset()
        self.gru.reset()
        self.conv_head.reset()

    def forward(self, net, inp, corr, depths, poses, attn=None):
        motion_features = self.encoder(depths, poses, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        
        delta = self.conv_head(net)

        return net, delta



