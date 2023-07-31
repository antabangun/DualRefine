import glob
import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
from multiprocessing import Process, Queue

import torch
from torchvision import transforms

from dualrefine import networks
from dualrefine.options import MonodepthOptions
from dualrefine.layers import disp_to_depth

import pdb


class imageViewer(object):
    def __init__(self, q, eval_thres):
        self.frame = q
        self.eval_thres = eval_thres

        self.stop_show = Queue()
        self.show_thread = Process(target=self.show)
        self.show_thread.start()

    def stop(self):
        self.stop_show.put(True)
        self.show_thread.join()
        print('imageViewer stopped...')

    def show(self):
        while self.stop_show.empty():
            if (not self.frame.empty()):
                while not self.frame.empty():
                    frame = self.frame.get()

                cv2.imshow('frame', frame)
                k = cv2.waitKey(1)
                if k == ord('w'):
                    self.eval_thres.put(1)
                elif k == ord('s'):
                    self.eval_thres.put(-1)


class DepthEstimator(object):
    def __init__(self, opt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        depth_pose_path = os.path.join(opt.load_weights_folder, "depth_pose.pth")
        depth_pose_class = networks.DEQDepthPose

        depth_pose_dict = torch.load(depth_pose_path)

        min_depth_bin = depth_pose_dict.get('min_depth_bin')
        max_depth_bin = depth_pose_dict.get('max_depth_bin')
        mean_depth_bin = depth_pose_dict.get('mean_depth_bin')
        median_depth_bin = depth_pose_dict.get('median_depth_bin')
        try:
            HEIGHT, WIDTH = depth_pose_dict['height'], depth_pose_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the depth_pose state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        self.HEIGHT, self.WIDTH = HEIGHT, WIDTH

        # setup models
        depth_pose_opts = dict(args=opt)
        depth_pose = depth_pose_class(**depth_pose_opts)

        model_dict = depth_pose.state_dict()
        depth_pose.load_state_dict({k: v for k, v in depth_pose_dict.items() if k in model_dict})

        depth_pose.eval()
        depth_pose.update_depth_bins(
            max_depth_bin, min_depth_bin, mean_depth_bin, median_depth_bin
        )

        if torch.cuda.is_available():
            depth_pose.cuda()

        self.depth_pose = depth_pose
        print("-> Models loaded!")
        self.to_tensor = transforms.ToTensor()

        # Intrinsic camera parameters
        K = np.array([[0.58, 0, 0.5, 0],
                        [0, 1.92, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]).astype(np.float32)
        K = torch.tensor(K, device=self.device)[None]
        self.intrinsics = dict()
        for scale in range(4):
            Kc = K.clone()
            Kc[:, 0, :] *= WIDTH // 2**scale
            Kc[:, 1, :] *= HEIGHT // 2**scale
            inv_Kc = torch.inverse(Kc)
            self.intrinsics[('K', scale)] = Kc
            self.intrinsics[('inv_K', scale)] = inv_Kc
        self.K0 = self.intrinsics[('K', 0)].clone().detach().cpu().numpy()[0]

        # setup image and pointcloud viewer
        self.im_queue, self.eval_thres = Queue(), Queue()
        self.image_viewer = imageViewer(q=self.im_queue, eval_thres=self.eval_thres)

    def callback(self, img0, img1):
        img0_resized = cv2.resize(img0, (self.WIDTH, self.HEIGHT))
        img1_resized = cv2.resize(img1, (self.WIDTH, self.HEIGHT))
        img0_tensor = self.to_tensor(img0_resized)[None].to(self.device)
        img1_tensor = self.to_tensor(img1_resized)[None].to(self.device)

        inp_data = [
            img1_tensor,
            img0_tensor,
            img0_tensor.new_zeros(img0_tensor.shape[0])]
        with torch.no_grad():
            disp_output, poses = self.depth_pose(inp_data, self.intrinsics, True)
        
        disp = disp_output[("disp", 0, 1)][0, 0]
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        disp_np, depth_np = disp.data.cpu().numpy(), depth.data.cpu().numpy()
        poses = poses[1][0].float().data.cpu().numpy()
        
        disp_color = cv2.applyColorMap(
            (200*disp_np).astype(np.uint8), cv2.COLORMAP_MAGMA)
        out_im = np.concatenate((
            img1_resized, disp_color), 0)
    
        self.im_queue.put(out_im)
        if not self.image_viewer.eval_thres.empty():
            self.depth_pose.eval_f_thres = \
                max(0, self.depth_pose.eval_f_thres + self.image_viewer.eval_thres.get())

        print("  -> # iters: {}".format(self.depth_pose.eval_f_thres))
        print()


def main(opt):
    sequence_path = os.path.join(
        opt.data_path, opt.sequence_date,
        f"{opt.sequence_date}_drive_{opt.sequence_id}_sync", "image_02", "data")
    image_paths = sorted(glob.glob(os.path.join(sequence_path, "*.png" if opt.png else "*.jpg")))
    
    depth_estimator = DepthEstimator(opt)

    image_paths_1 = image_paths[1:]
    for i, image_path in enumerate(image_paths_1):
        print(f"Processing image {i+1}/{len(image_paths)}")
        img0 = cv2.imread(image_paths[i])
        img1 = cv2.imread(image_path)
        depth_estimator.callback(img0, img1)

    depth_estimator.image_viewer.stop()


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument("--sequence_date", type=str,
                                default="2011_09_26",
                                help="path to the KITTI sequence")
    options.parser.add_argument("--sequence_id", type=str,
                                default="0009",
                                help="path to the KITTI sequence")
    main(options.parse())

