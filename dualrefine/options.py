import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
     def __init__(self):
          self.parser = argparse.ArgumentParser(description="DualRefine options")

          # PATHS
          self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="kitti_data")
          self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

          # TRAINING options
          self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
          self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
          self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18)
          self.parser.add_argument("--depth_binning",
                                 help="defines how the depth bins are constructed for the cost"
                                      "volume. 'linear' is uniformly sampled in depth space,"
                                      "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse'],
                                 default='linear'),
          self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=96)
          self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom"])
          self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
          self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
          self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
          self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
          self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
          self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
          self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
          self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
          
          self.parser.add_argument("--Tstar_D0_pair",
                                 action='store_true',
                                 help="If set, Tstar is paired with D0 for loss computation. \
                                           Otherwise, default is Tstar paired with Dstar")
          self.parser.add_argument("--Dstar_T0_pair",
                                 action='store_true',
                                 help="If set, Dstar is paired with T0 for loss computation. \
                                              Otherwise, default is Dstar paired with Tstar")
          self.parser.add_argument('--Pose_for_consistency_mask',
                                   type=str, default='Tstar', choices=['Tstar', 'T0'],
                                   help='Which pose to use for computing the consistency mask')
          self.parser.add_argument('--combine_via_mask', action='store_true',
                                   help='If set, combine the final depth maps via consistency mask')

          self.parser.add_argument('--mixed_precision',
                                   action='store_true', help='use mixed precision')

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
          self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
          self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
          self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
          self.parser.add_argument("--freeze_teacher_and_pose",
                                 action="store_true",
                                 help="If set, freeze the weights of the single frame teacher"
                                      " network and pose network.")
          self.parser.add_argument("--freeze_teacher_epoch",
                                 type=int,
                                 default=15,
                                 help="Sets the epoch number at which to freeze the teacher"
                                      "network and the pose network.")
          self.parser.add_argument("--unfreeze_student_epoch",
                                 type=int,
                                 default=15,
                                 help="Sets the epoch number at which to unfreeze the update student network.")
          self.parser.add_argument("--freeze_teacher_step",
                                 type=int,
                                 default=-1,
                                 help="Sets the step number at which to freeze the teacher"
                                      "network and the pose network. By default is -1 and so"
                                      "will not be used.")
          self.parser.add_argument("--pytorch_random_seed",
                                 default=None,
                                 type=int)

          # Cost volume options
          self.parser.add_argument('--num_cost_volume_head',
                                 help='The number of group/head for building the cost volume',
                                 type=int,
                                 default=1)
          self.parser.add_argument("--use_depth_bins_for_masking",
                                 action='store_true',
                                 help="If set, compute cost volume mask based on depth bins")
          
          self.parser.add_argument('--gap_factor',
                                   type=str, default='depth', choices=['depth', 'minmax'],
                                   help='gap factor for the epipolar sampling. \
                                             Default is "depth", which means the sampling radius is multiplied by the depth. \
                                             If set to "minmax", the gap factor is computed based on the computed min and max depth')
          self.parser.add_argument('--gap_factor_depth_ratio',
                                   type=int, default=8,
                                   help='if gap_factor is None, use this ratio to compute the gap factor')

          # Pose updates options
          self.parser.add_argument("--disable_pose_updates",
                                 action='store_true',
                                 help="If set, pose updates are disabled.")
                    
          self.parser.add_argument('--num_pose_iter',
                                   type=int, default=1,
                                   help='number of pose iterations at each update step')
          
          self.parser.add_argument('--disable_evolving_pose_weight',
                                   action='store_true',
                                   help='Disable computation of weight for the pose update at every time step')
          self.parser.add_argument('--disable_fixed_pose_weight',
                                   action='store_true',
                                   help='Disable computation of fixed weight for the pose update')
          self.parser.add_argument('--robust_pose_loss',
                                   action='store_true',
                                   help='Use robust pose loss')
          
          self.parser.add_argument('--num_levels',
                                   type=int, default=3,
                                   help="number of levels for correlation lookup.")
          self.parser.add_argument('--corr_radius',
                                   type=int, default=8,
                                   help="Radius size of the correlation lookup.")
          
          # DEQ options
          self.parser.add_argument('--disable_wnorm',
                                   action='store_true',
                                   help="use weight normalization")
          self.parser.add_argument('--f_solver',
                                   default='anderson', type=str, choices=['anderson', 'broyden'],
                                   help='forward solver to use (only anderson supported now)')
          self.parser.add_argument('--b_solver',
                                   default='broyden', type=str, choices=['anderson', 'broyden'],
                                   help='backward solver to use')
          self.parser.add_argument('--f_thres',
                                   type=int, default=6,
                                   help='forward pass solver threshold')
          self.parser.add_argument('--b_thres',
                                   type=int, default=6,
                                   help='backward pass solver threshold')
          self.parser.add_argument('--stop_mode',
                                   type=str, default="abs",
                                   help="fixed-point convergence stop mode")
          self.parser.add_argument('--eval_factor',
                                   type=float, default=1.,
                                   help="factor to scale up the f_thres at test for better precision.")

          self.parser.add_argument('--ift',
                                   action='store_true',
                                   help="use implicit differentiation.")
          self.parser.add_argument('--safe_ift',
                                   action='store_true',
                                   help="use a safer function for IFT to avoid segment fault.")
          self.parser.add_argument('--n_losses',
                                   type=int, default=1,
                                   help="number of loss terms (uniform spaced, 1 + fixed point correction).")
          self.parser.add_argument('--indexing',
                                   type=int, nargs='+', default=[],
                                   help="indexing for fixed point correction.")
          self.parser.add_argument('--sup_all',
                                   action='store_true',
                                   help="supervise all the trajectories by Phantom Grad.")
          self.parser.add_argument('--phantom_grad',
                                   type=int, nargs='+', default=[1],
                                   help="steps of Phantom Grad")
          self.parser.add_argument('--tau',
                                   type=float, default=1.0,
                                   help="damping factor for unrolled Phantom Grad")

          # Monodepth2/Manydepth options
          self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
          self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
          self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
          self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
          self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
          self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
          self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
          self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
          self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training")
                                      
          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
          self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

          # LOADING options
          self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
          self.parser.add_argument("--mono_weights_folder",
                                 type=str)
          self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
          self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
          self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')

          # EVALUATION options
          self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
          self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
          self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=["eigen", "eigen_benchmark", "benchmark",
                                          "odom_9", "odom_10"],
                                 help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
          self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
          self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
          self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
          self.parser.add_argument("--zero_cost_volume",
                                 action="store_true",
                                 help="If set, during evaluation all poses will be set to 0, and "
                                      "so we will evaluate the model in single frame mode")
          self.parser.add_argument('--static_camera',
                                 action='store_true',
                                 help='If set, during evaluation the current frame will also be'
                                      'used as the lookup frame, to simulate a static camera')
          self.parser.add_argument('--eval_teacher',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')

          self.parser.add_argument('--debug',
                                   action='store_true',
                                   help='If set, will run in debug mode')

     def parse(self):
        self.options = self.parser.parse_args()
        return self.options
