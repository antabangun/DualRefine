from dis import dis
import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import cv2
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from .utils import readlines, sec_to_hm_str
from .layers import SSIM, BackprojectDepth, Project3D, \
    disp_to_depth, get_smooth_loss, compute_depth_errors

from dualrefine import datasets, networks
from dualrefine.networks.utils.utils import autocast
import matplotlib.pyplot as plt

import pdb


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        print('using adaptive depth binning!')
        self.min_depth_tracker = 0.1
        self.max_depth_tracker = 10.0
        self.mean_depth_tracker = 1.0
        self.median_depth_tracker = 1.0
        self.adapt_depth = True

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["depth_pose"] = networks.DEQDepthPose(self.opt)
        self.models["depth_pose"].to(self.device)
        self.parameters_to_train += list(self.models["depth_pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        self.batch_size = self.opt.batch_size

        train_dataset = self.dataset(
            self.opt,
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)
        val_dataset = self.dataset(
            self.opt,
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        num_train_samples = len(train_filenames)
        self.num_epochs = self.opt.num_epochs + 5 if self.opt.f_thres > 0 else self.opt.num_epochs
        self.num_total_steps = num_train_samples // self.batch_size * self.num_epochs

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales + list(range(self.opt.n_losses)):
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        if self.opt.mixed_precision:
            self.scaler = GradScaler()

        self.f_thres = self.opt.f_thres
        assert self.f_thres > 0, "f_thres should be greater than 0"

        self.autocast = self.opt.mixed_precision
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.num_epochs):
            if self.epoch == self.opt.freeze_teacher_epoch:
                self.models["depth_pose"].freeze_backbone()
                self.adapt_depth = False

            if self.epoch == self.opt.unfreeze_student_epoch:
                self.models["depth_pose"].unfreeze_student()
                self.f_thres = self.opt.f_thres

            current_lr = self.model_lr_scheduler.get_last_lr()
            print(f"Current learning rate: {current_lr}")
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        num_iters = len(self.train_loader)
        
        iter_train_loader = iter(self.train_loader)
        for batch_idx in range(num_iters):

            before_op_time = time.time()
            inputs = next(iter_train_loader)
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.opt.mixed_precision:
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(self.model_optimizer)
                torch.nn.utils.clip_grad_norm_(self.models["depth_pose"].parameters(), 0.1)
                
                self.scaler.step(self.model_optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            if self.step == self.opt.freeze_teacher_step:
                self.freeze_teacher()

            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if isinstance(ipt, torch.Tensor):
                inputs[key] = ipt.to(self.device)

        self.models["depth_pose"].update_depth_bins(
            self.max_depth_tracker, self.min_depth_tracker, self.mean_depth_tracker, self.median_depth_tracker
        )
        outputs = {}

        height, width = self.opt.height, self.opt.width

        imgs = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            img_inputs = [imgs[0], imgs[f_i]]

            img_inputs.append(imgs[0].new_zeros(imgs[0].shape[0], dtype=int))
                
            if f_i < 0:
                disps, poses = self.models["depth_pose"](
                    img_inputs, inputs, True)
                if self.opt.mixed_precision:
                    for key, val in disps.items():
                        disps[key] = val.float()
                outputs.update(disps)

                edge_mask = F.interpolate(
                    self.models["depth_pose"].edge_mask.type(torch.float),
                    [height, width],
                    mode="nearest")
                outputs["edge_mask"] = edge_mask.clone()
                outputs["consistency_mask"] = 1  # edge_mask.clone()
                if not self.opt.disable_motion_masking:
                    mask = F.interpolate(
                        self.models["depth_pose"].mask.type(torch.float),
                        [height, width],
                        mode="nearest")
                    outputs["consistency_mask"] *= mask.clone()
                    outputs["mask"] = mask.clone()

                    max_d = F.interpolate(
                        self.models["depth_pose"].max_d.type(torch.float),
                        [height, width],
                        mode="nearest")
                    outputs["max_d"] = max_d.clone()

                outputs[("cam_T_cam", 0, f_i)] = poses[0].float()
                outputs[("cam_T_cam", 0, f_i, 1)] = poses[1].float()

            elif f_i > 0:
                poses = self.models["depth_pose"].forward_pose(
                    img_inputs, False)

                outputs[("cam_T_cam", 0, f_i)] = poses.float()

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        # loss for pose updates
        if not self.opt.disable_pose_updates:
            self.pose_update_generate_images_pred(inputs, outputs)
            pose_update_losses = self.compute_pose_update_losses(inputs, outputs)
            for key, val in pose_update_losses.items():
                if key in losses.keys():
                    losses[key] += val
                else:
                    losses[key] = val

        # update adaptive depth bins
        if self.adapt_depth:
            self.update_adaptive_depth_bins(outputs)

        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        B, _, H, W = outputs[("disp", 0, 0)].shape

        min_depth = outputs[('depth', 0, 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('depth', 0, 0, 0)].detach().max(-1)[0].max(-1)[0]
        mean_depth = outputs[('depth', 0, 0, 0)].detach().mean(-1)[0].mean(-1)[0]
        median_depth = outputs[('depth', 0, 0, 0)].detach().median(-1)[0].median(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()
        mean_depth = mean_depth.mean().cpu().item()
        median_depth = median_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01
        self.mean_depth_tracker = self.mean_depth_tracker * 0.99 + mean_depth * 0.01
        self.median_depth_tracker = self.median_depth_tracker * 0.99 + median_depth * 0.01

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        height, width = self.opt.height, self.opt.width
        backproject_depth = self.backproject_depth
        project_3d = self.project_3d

        for scale in self.opt.scales:
            n_losses = self.opt.n_losses+1 if scale in [0, 1, 2] else 1
            for deq_iter in range(n_losses):
                if scale == 1:
                    continue
                disp = outputs[("disp", scale, deq_iter)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
                        disp, [height, width], mode="bilinear", align_corners=False)
                    source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, scale, deq_iter)] = depth

                for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                    if frame_id == 1:
                        T = outputs[("cam_T_cam", 0, frame_id)]
                        if deq_iter > 0:
                            T = T.detach()

                    else:
                        if deq_iter > 0:
                            if self.opt.Dstar_T0_pair:
                                # don't update posenet based on multi frame prediction
                                T = outputs[("cam_T_cam", 0, frame_id)]
                                T = T.detach()
                            else:
                                T = outputs[("cam_T_cam", 0, frame_id, 1)]
                        else:
                            T = outputs[("cam_T_cam", 0, frame_id)]

                    assert source_scale == 0
                    cam_points = backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                    outputs[("sample", frame_id, scale, deq_iter)] = pix_coords

                    outputs[("color", frame_id, scale, deq_iter)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale, deq_iter)],
                        padding_mode="border", align_corners=False)

                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale, deq_iter)] = \
                            inputs[("color", frame_id, source_scale)]

    def pose_update_generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        backproject_depth = self.backproject_depth
        project_3d = self.project_3d
        source_scale = 0
        if self.opt.Tstar_D0_pair:
            # don't update depthnet based on multi frame prediction
            depth = outputs[("depth", 0, 0, 0)].clone().detach()
        else:
            depth = outputs[("depth", 0, 0, self.opt.n_losses)]

        T = outputs[("cam_T_cam", 0, -1, 1)]

        cam_points = backproject_depth[source_scale](
            depth, inputs[("inv_K", source_scale)])
        pix_coords = project_3d[source_scale](
            cam_points, inputs[("K", source_scale)], T)

        outputs[("color", -1, 0, 0, 1)] = F.grid_sample(
            inputs[("color", -1, source_scale)],
            pix_coords,
            padding_mode="border", align_corners=False)
        

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            if self.f_thres > 0:
                n_losses = self.opt.n_losses+1 if scale in [0, 1, 2] else 1
                for deq_iter in range(n_losses):
                    if scale == 1:
                        continue
                    reprojection_losses = []

                    disp = outputs[("disp", scale, deq_iter)]
                    color = inputs[("color", 0, scale)]
                    target = inputs[("color", 0, source_scale)]

                    for frame_id in self.opt.frame_ids[1:]:
                        pred = outputs[("color", frame_id, scale, deq_iter)]
                        reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                    reprojection_losses = torch.cat(reprojection_losses, 1)

                    if not self.opt.disable_automasking:
                        identity_reprojection_losses = []
                        for frame_id in self.opt.frame_ids[1:]:
                            pred = inputs[("color", frame_id, source_scale)]
                            identity_reprojection_losses.append(
                                self.compute_reprojection_loss(pred, target))

                        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                        if self.opt.avg_reprojection:
                            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                        else:
                            # differently to Monodepth2, compute mins as we go
                            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                                    keepdim=True)
                    else:
                        identity_reprojection_loss = None

                    if self.opt.avg_reprojection:
                        reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                    else:
                        # differently to Monodepth2, compute mins as we go
                        reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

                    if not self.opt.disable_automasking:
                        # add random numbers to break ties
                        identity_reprojection_loss += torch.randn(
                            identity_reprojection_loss.shape).to(self.device) * 0.00001

                    # find minimum losses from [reprojection, identity]
                    reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                                    identity_reprojection_loss)

                    # find which pixels to apply reprojection loss to, and which pixels to apply
                    # consistency loss to
                    if deq_iter > 0:  # and scale == 2:
                        if not self.opt.disable_motion_masking:
                            reprojection_loss_mask = (reprojection_loss_mask *
                                                    outputs['consistency_mask'])
                        consistency_mask = (1 - reprojection_loss_mask).float()

                    # standard reprojection loss
                    reprojection_loss = reprojection_loss * reprojection_loss_mask
                    reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

                    # consistency loss:
                    if deq_iter > 0:  # and scale == 2:
                        multi_depth = outputs[("depth", 0, scale, deq_iter)]
                        # no gradients for mono prediction!
                        mono_depth = outputs[("depth", 0, scale, 0)].detach()
                        consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                        consistency_loss = consistency_loss.mean()
                        
                        # save for logging to tensorboard
                        consistency_target = (mono_depth.detach() * consistency_mask +
                                            multi_depth.detach() * (1 - consistency_mask))
                        consistency_target = 1 / consistency_target
                        outputs["consistency_target/{}_{}".format(scale, deq_iter)] = consistency_target
                        losses['consistency_loss/{}_{}'.format(scale, deq_iter)] = consistency_loss
                    else:
                        consistency_loss = 0

                    losses['reproj_loss/{}'.format(scale)] = reprojection_loss

                    loss += reprojection_loss + consistency_loss

                    mean_disp = disp.mean(2, True).mean(3, True)
                    norm_disp = disp / (mean_disp + 1e-7)
                    smooth_loss = get_smooth_loss(norm_disp, color)

                    loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                    total_loss += loss
                    losses["loss/{}_{}".format(scale, deq_iter)] = loss
            else:
                reprojection_losses = []

                disp = outputs[("disp", scale, 0)]
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale, 0)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # differently to Monodepth2, compute mins as we go
                        identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                                keepdim=True)
                else:
                    identity_reprojection_loss = None

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).to(self.device) * 0.00001

                # find minimum losses from [reprojection, identity]
                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                                identity_reprojection_loss)

                # standard reprojection loss
                reprojection_loss = reprojection_loss * reprojection_loss_mask
                reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

                losses['reproj_loss/{}'.format(scale)] = reprojection_loss

                loss += reprojection_loss

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss += loss
                losses["loss/{}_{}".format(scale, 0)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_pose_update_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        source_scale = 0

        reprojection_losses = []

        color = inputs[("color", 0, 0)]
        target = inputs[("color", 0, 0)]

        for frame_id in self.opt.frame_ids[1:]:
            pred = outputs[("color", -1, 0, 0, 1)] if frame_id == -1 else outputs[("color", frame_id, 0, 0)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))
        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not self.opt.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                        keepdim=True)
        else:
            identity_reprojection_loss = None

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            # differently to Monodepth2, compute mins as we go
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).to(self.device) * 0.00001

        # find minimum losses from [reprojection, identity]
        reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                        identity_reprojection_loss)

        # standard reprojection loss
        reprojection_loss = reprojection_loss * reprojection_loss_mask
        reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

        ##########################

        losses['reproj_loss/pose_{}'.format(0)] = reprojection_loss

        loss += reprojection_loss

        total_loss += loss
        losses["loss/pose_{}_{}".format(0, 0)] = loss

        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        writer.add_scalar("max_depth", self.max_depth_tracker, self.step)
        writer.add_scalar("min_depth", self.min_depth_tracker, self.step)
        writer.add_scalar("mean_depth", self.mean_depth_tracker, self.step)
        writer.add_scalar("median_depth", self.median_depth_tracker, self.step)
        
        disp_0_hist = np.histogram(
            outputs['disp', 0, 0].data.cpu().numpy(),
            bins=10*self.opt.num_depth_bins,
            range=(self.opt.min_depth, self.opt.max_depth))
        disp_n_hist = np.histogram(
            outputs['disp', 0, self.opt.n_losses].data.cpu().numpy(),
            bins=10*self.opt.num_depth_bins,
            range=(self.opt.min_depth, self.opt.max_depth))
        writer.add_histogram("disp_0_hist", disp_0_hist[0], self.step)
        writer.add_histogram("disp_n_hist", disp_n_hist[0], self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s, self.opt.n_losses)][j].data, self.step)

            disp = colormap(outputs[("disp", s, self.opt.n_losses)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('disp', s, 0)][j, 0])
            writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)
            
            mask = colormap(outputs['mask'][j, 0])
            writer.add_image(
                "mask/{}".format(j),
                mask, self.step)
            edge_mask = colormap(outputs['edge_mask'][j, 0])
            writer.add_image(
                "edge_mask/{}".format(j),
                edge_mask, self.step)
            max_d = colormap(outputs['max_d'][j, 0])
            writer.add_image(
                "max_d/{}".format(j),
                max_d, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            # save the sizes - these are needed at prediction time
            to_save['height'] = self.opt.height
            to_save['width'] = self.opt.width
            # save estimates of depth bins
            to_save['min_depth_bin'] = self.min_depth_tracker
            to_save['max_depth_bin'] = self.max_depth_tracker
            to_save['mean_depth_bin'] = self.mean_depth_tracker
            to_save['median_depth_bin'] = self.median_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        ############################
        # Teacher weights
        load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            min_depth_bin = pretrained_dict.get('min_depth_bin')
            max_depth_bin = pretrained_dict.get('max_depth_bin')
            mean_depth_bin = pretrained_dict.get('mean_depth_bin')
            median_depth_bin = pretrained_dict.get('median_depth_bin')
            print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
            if min_depth_bin is not None:
                self.min_depth_tracker = min_depth_bin
                self.max_depth_tracker = max_depth_bin
                self.mean_depth_tracker = mean_depth_bin
                self.median_depth_tracker = median_depth_bin

            load_dict = {}
            skip = []  # ['pose_update', 'hidden', 'context', 'conv2']
            for k, v in pretrained_dict.items():
                if k in model_dict and k.split('.')[0] not in skip:
                    load_dict[k] = v
            pretrained_dict = load_dict

            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
