import os
import logging
import numpy as np
import imageio
import time
import math
import yaml
import copy
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from fast_pytorch_kmeans import KMeans
import torch
import torch.nn.functional as F
from models.nerf import get_embedder, NeRF
from models.rays import sampling_index, sample_pdf, create_rays
from training.training_utils import batchify_rays, calculate_segmentation_metrics
from models.model_utils import raw2outputs
from models.model_utils import run_network
from visualisation.tensorboard_vis import TFVisualizer
from utils.geometry_utils import back_project
from tqdm import tqdm, trange
from imgviz import label_colormap, depth2rgb
from scipy import ndimage
from aggregation import aggregator_trainer
from scipy.special import softmax
to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def select_gpus(gpus):
    """
    takes in a string containing a comma-separated list
    of gpus to make visible to tensorflow, e.g. '0,1,3'
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus is not '':
        logging.info("Using gpu's: {}".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        logging.info('Using all available gpus')

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.set_params()

        self.training = True  # training mode by default
        # create tfb Summary writers and folders
        tf_log_dir = os.path.join(config["experiment"]["save_dir"], "tfb_logs")
        if not os.path.exists(tf_log_dir):
            os.makedirs(tf_log_dir)
        self.tfb_viz = TFVisualizer(tf_log_dir, config["logging"]["step_log_tfb"], config)
        self.cpu = config["experiment"]["load_on_cpu"]
        self.pos_threshold = config["experiment"]["pos_threshold"]
        self.nocs_pos_threshold = config["experiment"]["nocs_pos_threshold"]
            
    def save_config(self):
        # save config to save_dir for the convience of checking config later
        with open(os.path.join(self.config["experiment"]["save_dir"], 'exp_config.yaml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def set_params_replica(self):
        self.H = self.config["experiment"]["height"]
        self.W = self.config["experiment"]["width"]

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W/self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = self.config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self.config["render"]["test_viz_factor"])
        self.H_scaled = self.H//self.test_viz_factor
        self.W_scaled = self.W//self.test_viz_factor
        self.fx_scaled = self.W_scaled / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy_scaled = self.fx_scaled
        self.cx_scaled = (self.W_scaled - 1.0) / 2.0
        self.cy_scaled = (self.H_scaled - 1.0) / 2.0

        self.K = np.zeros((3, 3))
        self.K[0, 0] = self.fx_scaled
        self.K[1, 1] = self.fy_scaled
        self.K[0, -1] = self.cx
        self.K[1, -1] = self.cy
        self.K[-1, -1] = 1

        self.save_config()

    def set_params_scannet(self, data):
        self.H = self.config["experiment"]["height"]
        self.W = self.config["experiment"]["width"]
        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W/self.H

        K = data.intrinsics
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, -1]
        self.cy = K[1, -1]
        self.near, self.far = self.config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self.config["render"]["test_viz_factor"])
        self.H_scaled = self.config["experiment"]["height"]//self.test_viz_factor
        self.W_scaled = self.config["experiment"]["width"]//self.test_viz_factor
        self.fx_scaled = self.fx/self.test_viz_factor
        self.fy_scaled = self.fy/self.test_viz_factor
        self.cx_scaled = (self.W_scaled - 0.5) / 2.0
        self.cy_scaled = (self.H_scaled - 0.5) / 2.0

        self.K = np.zeros((3, 3))
        self.K[0, 0] = self.fx_scaled
        self.K[1, 1] = self.fy_scaled
        self.K[0, -1] = self.cx
        self.K[1, -1] = self.cy
        self.K[-1, -1] = 1

        self.save_config()

    def set_params(self):
        self.load_dino = self.config["experiment"]["load_dino"]
        self.load_lseg = self.config["experiment"]["load_lseg"]
        self.fea_dim = self.config["experiment"]["feature_dim"]
        if self.load_lseg:
            self.fea_dim = 512

        #render options
        self.n_rays = eval(self.config["render"]["N_rays"])  if isinstance(self.config["render"]["N_rays"], str) \
            else self.config["render"]["N_rays"]

        self.N_samples = self.config["render"]["N_samples"]
        self.netchunk = eval(self.config["model"]["netchunk"]) if isinstance(self.config["model"]["netchunk"], str) \
            else self.config["model"]["netchunk"]

        self.chunk = eval(self.config["model"]["chunk"])  if isinstance(self.config["model"]["chunk"], str) \
            else self.config["model"]["chunk"]

        self.use_viewdir = self.config["render"]["use_viewdirs"]

        self.convention = self.config["experiment"]["convention"]

        self.endpoint_feat = self.config["experiment"]["endpoint_feat"] if "endpoint_feat" in self.config["experiment"].keys() else False

        self.N_importance = self.config["render"]["N_importance"]
        self.raw_noise_std = self.config["render"]["raw_noise_std"]
        self.white_bkgd = self.config["render"]["white_bkgd"]
        self.perturb = self.config["render"]["perturb"]

        self.no_batching = self.config["render"]["no_batching"]

        self.lrate = float(self.config["train"]["lrate"])
        self.lrate_decay = float(self.config["train"]["lrate_decay"])

        # logging
        self.save_dir = self.config["experiment"]["save_dir"]

        self.sample_semantic = self.config["experiment"]["sample_semantic"]

    def prepare_data_replica(self, data):
        self.ignore_label = -1

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples

        self.train_ids = data.train_ids
        self.test_ids = data.test_ids
        self.mask_ids = data.mask_ids.astype(np.bool)

        self.num_train = data.train_num
        self.num_test = data.test_num

        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including void class=0
        self.num_valid_semantic_class = self.num_semantic_class - 1  # exclude void class
        assert self.num_semantic_class==data.num_semantic_class

        total_num_classes = 101

        colour_map_np = label_colormap(total_num_classes)[data.semantic_classes] # select the existing class from total colour map
        self.colour_map = torch.from_numpy(colour_map_np)
        self.valid_colour_map = torch.from_numpy(colour_map_np[1:,:]) # exclude the first colour map to colourise rendered segmentation without void index

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(np.arange(self.num_semantic_class))

        #####training data#####
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(self.train_image.permute(0,3,1,2,), 
                                    scale_factor=1/self.config["render"]["test_viz_factor"], 
                                    mode='bilinear').permute(0,2,3,1)

        # vit descriptor
        self.train_vit = torch.from_numpy(train_samples["descriptor"])

        # depth
        self.train_depth = torch.from_numpy(train_samples["depth"])
        self.viz_train_depth = np.stack([depth2rgb(dep, min_value=self.near, max_value=self.far) for dep in train_samples["depth"]], axis=0) # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.train_depth_scaled = F.interpolate(torch.unsqueeze(self.train_depth, dim=1).float(), 
                                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                            mode='bilinear').squeeze(1).cpu().numpy()

        # semantic 
        self.train_semantic = torch.from_numpy(train_samples["semantic_remap"])
        self.viz_train_semantic = np.stack([colour_map_np[sem] for sem in self.train_semantic], axis=0) # [num_test, H, W, 3]
        self.train_semantic_scaled = F.interpolate(torch.unsqueeze(self.train_semantic, dim=1).float(),
                                                            scale_factor=1/self.config["render"]["test_viz_factor"],
                                                            mode='nearest').squeeze(1)
        self.train_semantic_scaled = self.train_semantic_scaled.cpu().numpy() - 1

        self.train_semantic_clean = torch.from_numpy(train_samples["semantic_remap_clean"])
        self.viz_train_semantic_clean = np.stack([colour_map_np[sem] for sem in self.train_semantic_clean], axis=0) # [num_test, H, W, 3]
        
        # process the clean label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(torch.unsqueeze(self.train_semantic_clean, dim=1).float(), 
                                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                            mode='nearest').squeeze(1)
        self.train_semantic_clean_scaled = self.train_semantic_clean_scaled.cpu().numpy() - 1 
        # pose 
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()


        #####test data#####
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(self.test_image.permute(0,3,1,2,), 
                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                            mode='bilinear').permute(0,2,3,1)


        # depth
        self.test_depth = torch.from_numpy(test_samples["depth"])  # [num_test, H, W]
        self.viz_test_depth = np.stack([depth2rgb(dep, min_value=self.near, max_value=self.far) for dep in test_samples["depth"]], axis=0) # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.test_depth_scaled = F.interpolate(torch.unsqueeze(self.test_depth, dim=1).float(), 
                                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                            mode='bilinear').squeeze(1).cpu().numpy()
        # semantic 
        self.test_semantic = torch.from_numpy(test_samples["semantic_remap"])  # [num_test, H, W]

        self.viz_test_semantic = np.stack([colour_map_np[sem] for sem in self.test_semantic], axis=0) # [num_test, H, W, 3]

        # we only add noise to training images, therefore test images are kept intact. No need for test_remap_clean
        # process the clean label for evaluation purpose
        self.test_semantic_scaled = F.interpolate(torch.unsqueeze(self.test_semantic, dim=1).float(), 
                                                    scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                    mode='nearest').squeeze(1)
        self.test_semantic_scaled = self.test_semantic_scaled.cpu().numpy() - 1 # shift void class from value 0 to -1, to match self.ignore_label
        # pose 
        self.test_Ts = torch.from_numpy(test_samples["T_wc"]).float()  # [num_test, 4, 4]

        # vit descriptor
        self.test_vit = torch.from_numpy(test_samples["descriptor"])

        if self.cpu is False:
            print("load on gpu!")
            self.train_image = self.train_image.cuda()
            self.train_image_scaled = self.train_image_scaled.cuda()
            self.train_depth = self.train_depth.cuda()
            self.train_semantic = self.train_semantic.cuda()
            self.train_vit = self.train_vit.cuda()

            self.test_image = self.test_image.cuda()
            self.test_image_scaled = self.test_image_scaled.cuda()
            self.test_depth = self.test_depth.cuda()
            self.test_semantic = self.test_semantic.cuda()
            self.colour_map = self.colour_map.cuda()
            self.valid_colour_map = self.valid_colour_map.cuda()
            self.test_vit = self.test_vit.cuda()

        # set the data sampling paras which need the number of training images
        if self.no_batching is False: # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train*self.H*self.W)

        # add datasets to tfboard for comparison to rendered images
        self.tfb_viz.tb_writer.add_image('Train/rgb_GT', train_samples["image"], 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/depth_GT', self.viz_train_depth, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/vis_sem_label_GT', self.viz_train_semantic, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/vis_sem_label_GT_clean', self.viz_train_semantic_clean, 0, dataformats='NHWC')

        # self.tfb_viz.tb_writer.add_image('Test/legend', np.expand_dims(legend_img_arr, axis=0), 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Test/rgb_GT', test_samples["image"], 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Test/depth_GT', self.viz_test_depth, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Test/vis_sem_label_GT', self.viz_test_semantic, 0, dataformats='NHWC')

    def prepare_data_scannet(self, data, gpu=True):
        self.ignore_label = -1

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples

        self.train_ids = data.train_ids
        self.test_ids = data.test_ids
        self.mask_ids = data.mask_ids

        self.num_train = data.train_num
        self.num_test = data.test_num

        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including void class=0
        self.num_valid_semantic_class = self.num_semantic_class - 1  # exclude void class ==0
        assert self.num_semantic_class==data.num_semantic_class

        self.num_instance = data.num_instance
        colour_map_np = data.colour_map_np_remap 
        self.colour_map = torch.from_numpy(colour_map_np)
        self.valid_colour_map  = torch.from_numpy(colour_map_np[1:,:]) # exclude the first colour map to colourise rendered segmentation without void index
        instance_colour_map_np = data.instance_colour_map_np
        self.instance_colour_map = torch.from_numpy(data.instance_colour_map_np)

        self.to_origin_transform = torch.tensor(data.to_origin_transform, dtype=torch.float32).cuda()
        self.extents = torch.tensor(data.extents, dtype=torch.float32).cuda()

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(np.arange(self.num_semantic_class))

        #####training data#####
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(self.train_image.permute(0,3,1,2,), 
                                    scale_factor=1/self.config["render"]["test_viz_factor"], 
                                    mode='bilinear').permute(0,2,3,1)

        # semantic 
        self.train_semantic = torch.from_numpy(train_samples["semantic_remap"])
        self.train_semantic_scaled = F.interpolate(torch.unsqueeze(self.train_semantic, dim=1).float(),
                                                            scale_factor=1/self.config["render"]["test_viz_factor"],
                                                            mode='nearest').squeeze(1)
        self.train_semantic_scaled = self.train_semantic_scaled.cpu().numpy() - 1
        self.viz_train_semantic = np.stack([colour_map_np[sem] for sem in self.train_semantic], axis=0) # [num_test, H, W, 3]

        self.train_semantic_clean = torch.from_numpy(train_samples["semantic_remap_clean"])
        self.viz_train_semantic_clean = np.stack([colour_map_np[sem] for sem in self.train_semantic_clean], axis=0) # [num_test, H, W, 3]

        # instance
        self.train_instance = torch.from_numpy(train_samples["instance"])
        self.viz_train_instance = np.stack([instance_colour_map_np[ins+1] for ins in self.train_instance], axis=0)
        self.train_instance_scaled = F.interpolate(torch.unsqueeze(self.train_instance, dim=1).float(),
                                                            scale_factor=1/self.config["render"]["test_viz_factor"],
                                                            mode='nearest').squeeze(1)
        self.train_instance_scaled = self.train_instance_scaled.cpu().numpy()
        
        # process the clean label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(torch.unsqueeze(self.train_semantic_clean, dim=1).float(), 
                                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                            mode='nearest').squeeze(1)
        self.train_semantic_clean_scaled = self.train_semantic_clean_scaled.cpu().numpy() - 1 
        # pose 
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()

        # vit descriptor
        self.train_vit = torch.from_numpy(train_samples["descriptor"])

        #####test data#####
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(self.test_image.permute(0,3,1,2,), 
                                            scale_factor=1/self.config["render"]["test_viz_factor"], 
                                            mode='bilinear').permute(0,2,3,1)

        # semantic 
        self.test_semantic = torch.from_numpy(test_samples["semantic_remap"])  # [num_test, H, W]
        self.viz_test_semantic = np.stack([colour_map_np[sem] for sem in self.test_semantic], axis=0) # [num_test, H, W, 3]

        # we do add noise only to training images used for training, test images are kept the same. No need for test_remap_clean

        # instance
        self.test_instance = torch.from_numpy(test_samples["instance"])
        self.viz_test_instance = np.stack([instance_colour_map_np[ins] for ins in self.test_instance], axis=0)
        self.test_instance_scaled = F.interpolate(torch.unsqueeze(self.test_instance, dim=1).float(),
                                                            scale_factor=1/self.config["render"]["test_viz_factor"],
                                                            mode='nearest').squeeze(1)
        self.test_instance_scaled = self.test_instance_scaled.cpu().numpy()

        # process the clean label for evaluation purpose
        self.test_semantic_scaled = F.interpolate(torch.unsqueeze(self.test_semantic, dim=1).float(), 
                                                    scale_factor=1/self.config["render"]["test_viz_factor"], 
                                                    mode='nearest').squeeze(1)
        self.test_semantic_scaled = self.test_semantic_scaled.cpu().numpy() - 1 # shift void class from value 0 to -1, to match self.ignore_label
        # pose 
        self.test_Ts = torch.from_numpy(test_samples["T_wc"]).float()  # [num_test, 4, 4]

        # vit descriptor
        self.test_vit = torch.from_numpy(test_samples["descriptor"])

        if self.cpu is False:
            self.train_image = self.train_image.cuda()
            self.train_image_scaled = self.train_image_scaled.cuda()
            # self.train_depth = self.train_depth.cuda()
            self.train_semantic = self.train_semantic.cuda()
            self.train_instance = self.train_instance.cuda()
            self.train_vit = self.train_vit.cuda()

            self.test_image = self.test_image.cuda()
            self.test_image_scaled = self.test_image_scaled.cuda()
            # self.test_depth = self.test_depth.cuda()
            self.test_semantic = self.test_semantic.cuda()
            self.test_instance = self.test_instance.cuda()
            self.test_vit = self.test_vit.cuda()
            self.colour_map = self.colour_map.cuda()
            self.valid_colour_map = self.valid_colour_map.cuda()
            self.instance_colour_map = self.instance_colour_map.cuda()


        # set the data sampling paras which need the number of training images
        if self.no_batching is False: # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train*self.H*self.W)

        # add datasets to tfboard for comparison to rendered images
        # self.tfb_viz.tb_writer.add_image('Train/legend', np.expand_dims(legend_img_arr, axis=0), 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/rgb_GT', train_samples["image"], 0, dataformats='NHWC')
        # self.tfb_viz.tb_writer.add_image('Train/depth_GT', self.viz_train_depth, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/vis_sem_label_GT', self.viz_train_semantic, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Train/vis_sem_label_GT_clean', self.viz_train_semantic_clean, 0, dataformats='NHWC')

        # self.tfb_viz.tb_writer.add_image('Test/legend', np.expand_dims(legend_img_arr, axis=0), 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Test/rgb_GT', test_samples["image"], 0, dataformats='NHWC')
        # self.tfb_viz.tb_writer.add_image('Test/depth_GT', self.viz_test_depth, 0, dataformats='NHWC')
        self.tfb_viz.tb_writer.add_image('Test/vis_sem_label_GT', self.viz_test_semantic, 0, dataformats='NHWC')

    def init_rays(self):
        # create rays
        rays = create_rays(self.num_train, self.train_Ts, self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                                self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)

        rays_vis = create_rays(self.num_train, self.train_Ts, self.H_scaled, self.W_scaled, self.fx_scaled, self.fy_scaled,
                            self.cx_scaled, self.cy_scaled, self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)

        rays_test = create_rays(self.num_test, self.test_Ts, self.H_scaled, self.W_scaled, self.fx_scaled, self.fy_scaled,
                                self.cx_scaled, self.cy_scaled, self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)

        # init rays
        self.rays = rays # [num_images, H*W, 11]
        self.rays_vis = rays_vis
        self.rays_test = rays_test
        if self.cpu is False:
            self.rays = self.rays.cuda()
            self.rays_vis = self.rays_vis.cuda()
            self.rays_test = self.rays_test.cuda()

    def perturb_one_direction(self, gradient, t=0.1, perturb_r=255, perturb_g=255, perturb_b=255):
        rgb_mat = torch.diag(torch.tensor([perturb_r, perturb_g, perturb_b])/255.0).double().cuda()
        gradient = rgb_mat @ gradient
        if self.config['experiment']['rgb_layer']:
            self.nerf_net_fine.rgb_linear.weight.data += gradient * t
        elif self.config["experiment"]["density_layer"]:
            self.nerf_net_fine.alpha_linear.weight.data += gradient * t
        elif self.config["experiment"]["pts_layer"]:
            self.nerf_net_fine.pts_linears[6].weight.data += gradient * t
        elif self.config["experiment"]["rgb_pts_layer"]:
            rgb_weight = self.nerf_net_fine.rgb_linear.weight.data
            rgb_weight_flat = rgb_weight.reshape(-1)
            self.nerf_net_fine.rgb_linear.weight.data += gradient[:rgb_weight_flat.shape[0]].reshape(rgb_weight.shape)
            pts_weight = self.nerf_net_fine.pts_linears[6].weight.data
            pts_weight_flat = pts_weight.reshape(-1)
            self.nerf_net_fine.pts_linears[6].weight.data += gradient[:pts_weight_flat.shape[0]].reshape(pts_weight.shape)
        else:
            raise NotImplementedError

    def render_gradient_rays(self, rays):
        if self.config["experiment"]["random_channel"]:
            rand_channel = np.random.randint(0, 3)
        grads = []
        with torch.enable_grad():
            output_dict = self.render_rays(rays)
            train_rgb_fine = output_dict["rgb_fine"]
            for idx, ray in enumerate(rays):
                self.clean_grad()
                if self.config["experiment"]["random_channel"]:
                    color = train_rgb_fine[idx, rand_channel]
                else:
                    color = train_rgb_fine[idx, ...].mean()
                if idx == rays.shape[0] -1 :
                    color.backward()
                else:
                    color.backward(retain_graph=True)
                grad = self.get_jacobian()
                grads.append(grad)
        grads = torch.stack(grads, 0)
        return grads

    def render_gradient_path(self, rays):
        grads_path = []
        for i, c2w in enumerate(tqdm(rays)):
            grads = self.render_gradient_rays(rays[i])
            grads_path.append(grads)
        grads_path = torch.stack(grads_path, 0)
        return grads_path

    def save_heatmap(self, idx, rgbs, save_path):
        # render test views and visualize feature difference / gradient difference?
        rgbs_p, disps_p, deps_p, vis_deps_p = self.render_path(self.rays_test, os.path.join(save_path, "perturbed_test_" + str(idx)), save_img=True)
        grays_p_flat = np.mean(rgbs_p.reshape((-1, 3)), -1)
        grays_flat = np.mean(rgbs.reshape((-1, 3)), -1)
        if self.config['experiment']['no_abs']:
            differences = grays_p_flat - grays_flat
        else:
            differences = np.abs(grays_p_flat - grays_flat)
        differences = differences.reshape((rgbs.shape[0], self.H_scaled, self.W_scaled))
        heatmaps = differences

        for idx_test, difference in enumerate(differences):
            # cv2.imwrite(os.path.join(self.save_dir, "perturbed_test_" + str(idx), str(idx_test) + "_difference.png"), difference)
            rgb_difference = ((rgbs * 255).astype(np.uint8))[idx_test, ...]
            rgb_difference[difference > 1e-3, :] = 0
            cv2.imwrite(
                os.path.join(save_path, "perturbed_test_" + str(idx), str(idx_test) + "_rgb_difference.png"),
                rgb_difference)
            plt.clf()
            heatmap = ndimage.filters.gaussian_filter(heatmaps[idx_test], 1, mode='nearest')
            sns.heatmap(heatmap)
            plt.savefig(os.path.join(save_path, "perturbed_test_" + str(idx), str(idx_test) + "_heatmap.png"))

    def perturb_gradient_render(self, rays, h, w, t=0.1, t_iter=None, perturb_r=255, perturb_g=255, perturb_b=255):
        with torch.no_grad():
            # sample random pixels from one image, render features, calculate gradients
            num_img, num_ray, ray_dim = rays.shape
            index_batch, index_hw = sampling_index(self.n_rays, num_img, h, w)
            if self.config["experiment"]["sample_fixed"]:
                # fixed
                index_batch = torch.tensor([[0]])
                index_hw = [int(i * num_ray / self.n_rays) for i in range(0, self.n_rays)]
                index_hw = torch.tensor(index_hw).reshape((1, self.n_rays))
            print("index_batch:", index_batch, "index_hw", index_hw)
            sampled_rays = rays[index_batch, index_hw, :]
            sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()

            # render and save perturbed ray
            rgbs_g, disps_g, deps_g, vis_deps_g = self.render_path(rays[index_batch, ...].reshape((1, num_ray, ray_dim)), save_dir=os.path.join(self.save_dir, "perturbed"), save_img=True)

            # render unperturbed test images
            rgbs, disps, deps, vis_deps = self.render_path(self.rays_test, os.path.join(self.save_dir, "unperturbed_test"), save_img=True)

        # sampled rays' grads
        grads = self.render_gradient_rays(sampled_rays)     # [sampled_chunk, param_num]
        print(grads.shape)

        with torch.no_grad():
            for idx, ray in enumerate(sampled_rays):
                # visualize perturbed ray
                perturb_ray_index = index_hw[0, idx].reshape((1, 1))
                print("perturb_ray_index:", perturb_ray_index)
                rgb_visualize_perturbed_flat = (rgbs_g*255).astype(np.uint8).reshape((-1, 3))
                rgb_visualize_perturbed_flat[perturb_ray_index, ...] = 0
                rgb_visualize_perturbed = rgb_visualize_perturbed_flat.reshape((self.H_scaled, self.W_scaled, 3))
                cv2.imwrite(os.path.join(self.save_dir, "perturbed", str(idx) + "_rgb_visualize_perturbed.png"), rgb_visualize_perturbed)
                visualize_perturbed_flat = np.ones(rgb_visualize_perturbed_flat.shape[0]) * 255
                visualize_perturbed_flat[perturb_ray_index] = 0
                visualize_perturbed = visualize_perturbed_flat.reshape((self.H_scaled, self.W_scaled))
                cv2.imwrite(os.path.join(self.save_dir, "perturbed", str(idx) + "_visualize_perturbed.png"), visualize_perturbed)

                if t_iter:
                    for t in range(100):
                        net_fine_copy = copy.deepcopy(self.nerf_net_fine)
                        net_coarse_copy = copy.deepcopy(self.nerf_net_coarse)
                        self.perturb_one_direction(grads[idx], t=t/100)
                        save_path = os.path.join(self.save_dir, 'gradient_'+str(t))
                        os.makedirs(save_path, exist_ok=True)
                        self.save_heatmap(idx, rgbs, save_path)
                        self.nerf_net_fine = net_fine_copy
                        self.nerf_net_coarse = net_coarse_copy
                else:
                    # perturb along the gradients
                    net_fine_copy = copy.deepcopy(self.nerf_net_fine)
                    net_coarse_copy = copy.deepcopy(self.nerf_net_coarse)
                    self.perturb_one_direction(grads[idx], t=t, perturb_r=perturb_r, perturb_g=perturb_g, perturb_b=perturb_b)
                    save_path = os.path.join(self.save_dir, 'gradient')
                    os.makedirs(save_path, exist_ok=True)
                    self.save_heatmap(idx, rgbs, save_path)
                    self.nerf_net_fine = net_fine_copy
                    self.nerf_net_coarse = net_coarse_copy

    def clean_grad(self):
        for param in self.nerf_net_coarse.parameters():
            if param.grad is not None:
                param.grad.zero_()
        for param in self.nerf_net_fine.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_jacobian(self, outputs=None, create_graph=True):
        if self.config["experiment"]["rgb_layer"]:
            # print(self.nerf_net_fine.rgb_linear.weight.requires_grad, self.nerf_net_fine.rgb_linear.weight.is_leaf)
            if outputs is not None:
                jacobian = torch.autograd.grad(outputs, self.nerf_net_fine.rgb_linear.weight, create_graph=create_graph)
            else:
                jacobian = self.nerf_net_fine.rgb_linear.weight.grad.data.double()
        elif self.config["experiment"]["density_layer"]:
            if outputs is not None:
                jacobian = torch.autograd.grad(outputs, self.nerf_net_fine.alpha_linear.weight, create_graph=create_graph)
            else:
                jacobian = self.nerf_net_fine.alpha_linear.weight.grad.double()
        elif self.config["experiment"]["all_para"]:
            jacobian = []
            for param in self.nerf_net_fine.parameters():
                if outputs is not None:
                    jacobian.append(torch.autograd.grad(outputs, param, create_graph=create_graph))
                else:
                    jacobian.append(param.grad.double())
            jacobian = torch.cat(jacobian)
        elif self.config["experiment"]["pts_layer"]:
            if outputs is not None:
                jacobian = torch.autograd.grad(outputs, self.nerf_net_fine.pts_linears[6].weight, create_graph=create_graph)
            else:
                jacobian = self.nerf_net_fine.pts_linears[6].weight.grad.double()
        elif self.config["experiment"]["rgb_pts_layer"]:
            jacobian = []
            if outputs is not None:
                jacobian.append(torch.tensor(torch.autograd.grad(outputs, self.nerf_net_fine.rgb_linear.weight, create_graph=create_graph)[0]).view(-1))
                jacobian.append(torch.tensor(torch.autograd.grad(outputs, self.nerf_net_fine.pts_linears[6].weight, create_graph=create_graph)[0]).view(-1))
            else:
                jacobian.append(self.nerf_net_fine.rgb_linear.weight.grad.double().view(-1))
                jacobian.append(self.nerf_net_fine.pts_linears[6].weight.grad.double().view(-1))
            jacobian = torch.cat(jacobian)
        else:
            raise NotImplementedError
        return jacobian

    def sample_data(self, step, rays, h, w, no_batching=True, mode="train", test_semantic=False, fixed=False, sample_pos=False, from_unmasked=False):
        # generate sampling index
        num_img, num_ray, ray_dim = rays.shape
        
        assert num_ray == h*w
        total_ray_num = num_img*h*w

        if mode == "train":
            image = self.train_image
            if self.load_dino or self.load_lseg:
                descriptor = self.train_vit
            sample_num = self.num_train
        elif mode == "test":
            image = self.test_image
            if test_semantic:
                # depth = self.test_depth
                semantic = self.test_semantic
            if self.load_dino or self.load_lseg:
                descriptor = self.test_vit
            sample_num = self.num_test
        elif mode == "vis":
            assert False
        else:
            assert False

        # sample rays and ground truth data
        sematic_available_flag = 1

        if no_batching or from_unmasked:  # sample random pixels from one random images
            if fixed:
                index_batch = torch.tensor([[0]])
                index_hw = [int(i * num_ray / self.n_rays) for i in range(0, self.n_rays)]
                index_hw = torch.tensor(index_hw).reshape((1, self.n_rays))
            elif from_unmasked:
                index_batch, index_hw = sampling_index(self.n_rays, self.mask_ids.sum(), h, w)
                index_batch = np.argwhere(self.mask_ids > 0)[index_batch]
            else:
                index_batch, index_hw = sampling_index(self.n_rays, num_img, h, w)
            sampled_rays = rays[index_batch, index_hw, :]

            flat_sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()
            gt_image = image.reshape(sample_num, -1, 3)[index_batch, index_hw, :].reshape(-1, 3)
            gt_image = gt_image.cuda()
            if test_semantic:
                # gt_depth = depth.reshape(sample_num, -1)[index_batch, index_hw].reshape(-1)
                # gt_depth = gt_depth.cuda()
                sematic_available_flag = self.mask_ids[index_batch] # semantic available if mask_id is 1 (train with rgb loss and semantic loss) else 0 (train with rgb loss only)
                gt_semantic = semantic.reshape(sample_num, -1)[index_batch, index_hw].reshape(-1)
                gt_semantic = gt_semantic.cuda()
            if self.load_dino or self.load_lseg:
                vit_descriptor = descriptor.reshape(sample_num, -1, self.fea_dim)[index_batch, index_hw, :].reshape(-1, self.fea_dim)
                vit_descriptor = vit_descriptor.cuda()
        else:  # sample from all random pixels
            index_hw = self.rand_idx[self.i_batch:self.i_batch+self.n_rays]

            flat_rays = rays.reshape([-1, ray_dim]).float()
            flat_sampled_rays = flat_rays[index_hw, :]
            gt_image = image.reshape(-1, 3)[index_hw, :]
            gt_image = gt_image.cuda()
            if test_semantic:
                # gt_depth = depth.reshape(-1)[index_hw]
                # gt_depth = gt_depth.cuda()
                gt_semantic = semantic.reshape(-1)[index_hw]
                gt_semantic = gt_semantic.cuda()
            if self.load_dino or self.load_lseg:
                vit_descriptor = descriptor.reshape(-1, self.fea_dim)[index_hw, :]
                vit_descriptor = vit_descriptor.cuda()

            self.i_batch += self.n_rays
            if self.i_batch >= total_ray_num:
                print("Shuffle data after an epoch!")
                self.rand_idx = torch.randperm(total_ray_num)
                self.i_batch = 0

        sampled_rays = flat_sampled_rays
        sampled_gt_rgb = gt_image
        if (self.load_dino or self.load_lseg) and test_semantic:
            sampled_vit = vit_descriptor
            # sampled_gt_depth = gt_depth

            sampled_gt_semantic = gt_semantic.long()  # required long type for nn.NLL or nn.crossentropy

            if sample_pos:
                if no_batching or from_unmasked:
                    return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag, sampled_vit, index_batch, index_hw
                return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag, sampled_vit, index_hw
            return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag, sampled_vit

        elif self.load_dino or self.load_lseg:
            sampled_vit = vit_descriptor
            if sample_pos:
                if no_batching or from_unmasked:
                    return sampled_rays, sampled_gt_rgb, sampled_vit, index_batch, index_hw
                return sampled_rays, sampled_gt_rgb, sampled_vit, index_hw
            return sampled_rays, sampled_gt_rgb, sampled_vit

        elif test_semantic:
            # sampled_gt_depth = gt_depth

            sampled_gt_semantic = gt_semantic.long()  # required long type for nn.NLL or nn.crossentropy

            if sample_pos:
                if no_batching or from_unmasked:
                    return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag, index_batch, index_hw
                return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag, index_hw
            return sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available_flag
        else:
            if sample_pos:
                if no_batching or from_unmasked:
                    return sampled_rays, sampled_gt_rgb, index_batch, index_hw
                return sampled_rays, sampled_gt_rgb, index_hw
            return sampled_rays, sampled_gt_rgb

    def render_rays(self, flat_rays, get_sampled_pts=False, test=False, chunk=None):
        """
        Render rays, run in optimisation loop
        Returns:
          List of:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          Dict of extras: dict with everything returned by render_rays().
        """
        if chunk is None:
            chunk = self.chunk

        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11

        # assert ray_shape[0] == self.n_rays  # this is not satisfied in test model
        fn = self.volumetric_rendering
        if get_sampled_pts:
            all_ret, all_pts, all_z_vals = batchify_rays(fn, flat_rays, chunk, self.cpu, get_sampled_pts=get_sampled_pts, test=test)
        else:
            all_ret = batchify_rays(fn, flat_rays, chunk, self.cpu, get_sampled_pts=get_sampled_pts, test=test)

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        if get_sampled_pts:
            return all_ret, all_pts, all_z_vals
        return all_ret

    def render_rays_propagate(self, flat_rays, tree, pts_label):
        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11

        fn = self.volumetric_rendering_propagate
        all_ret = []
        for i in tqdm(range(0, flat_rays.shape[0], self.chunk)):
            ret = fn(flat_rays[i:i + self.chunk].cuda(), tree, pts_label)
            all_ret.append(ret.cpu())
        all_ret = torch.cat(all_ret, 0)
        k_sh = list(ray_shape[:-1]) + list(all_ret.shape[1:])
        all_ret = torch.reshape(all_ret, k_sh)
        return all_ret

    def volumetric_rendering_propagate(self, ray_batch, tree, pts_label):
        N_rays = ray_batch.shape[0]
        if self.config['experiment']['test_instance']:
            num_class = self.num_instance
        else:
            num_class = self.num_valid_semantic_class

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals)  # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self.N_samples])

        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        raw_noise_std = self.raw_noise_std if self.training else 0
        raw_coarse = run_network(pts_coarse_sampled, viewdirs, self.nerf_net_coarse,
                                 self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)
        _, _, _, weights_coarse, _, _, _ = \
            raw2outputs(raw_coarse, z_vals, rays_d, raw_noise_std, self.white_bkgd,endpoint_feat=False)

        sem = np.zeros((N_rays, num_class))  # semantic logit

        if self.N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self.N_importance,
                                   det=(self.perturb == 0.) or (not self.training))
            z_samples = z_samples.detach()
            # detach so that grad doesn't propagate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            raw_fine = run_network(pts_fine_sampled, viewdirs, lambda x: self.nerf_net_fine(x, self.endpoint_feat), self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)

            raw_features = raw_fine[..., :3]
            if self.config["experiment"]["spatial"]:
                raw_features = torch.cat((raw_features, pts_fine_sampled), -1)
            elif self.config["experiment"]["spatial_embedding"]:
                raw_features = torch.cat((raw_features, self.embed_fn(pts_fine_sampled)), -1)

            difference_maps = []
            for class_idx, grad in enumerate(tree):
                if self.config['experiment']['mean_gradient']:
                    net_fine_copy = copy.deepcopy(self.nerf_net_fine)
                    net_coarse_copy = copy.deepcopy(self.nerf_net_coarse)
                    self.perturb_one_direction(grad, self.config["experiment"]["perturb_t"])
                    with torch.no_grad():
                        raw_fine = run_network(pts_fine_sampled, viewdirs, lambda x: self.nerf_net_fine(x, self.endpoint_feat), self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)
                    raw_features_p = raw_fine[..., :3]
                    if self.config['experiment']['no_abs']:
                        raw_features_diff = torch.mean(raw_features_p - raw_features, -1)
                    else:
                        raw_features_diff = torch.abs((torch.mean(raw_features_p - raw_features, -1)))
                    self.nerf_net_fine = net_fine_copy
                    self.nerf_net_coarse = net_coarse_copy
                elif self.config['experiment']['mean_response'] or self.config['experiment']['gradient_kmeans'] or self.config['experiment']['adaptive_selection']:
                    raw_features_diff = []
                    for g in grad:
                        net_fine_copy = copy.deepcopy(self.nerf_net_fine)
                        net_coarse_copy = copy.deepcopy(self.nerf_net_coarse)
                        self.perturb_one_direction(g, self.config["experiment"]["perturb_t"])
                        with torch.no_grad():
                            raw_fine = run_network(pts_fine_sampled, viewdirs,
                                                   lambda x: self.nerf_net_fine(x, self.endpoint_feat),
                                                   self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)
                        raw_features_p = raw_fine[..., :3]
                        if self.config['experiment']['no_abs']:
                            diff = torch.mean(raw_features_p - raw_features, -1)
                        else:
                            diff = torch.abs(torch.mean(raw_features_p - raw_features, -1))
                        raw_features_diff.append(diff)
                        if self.config['experiment']['merge_instance']:
                            difference_maps.append(diff)
                        self.nerf_net_fine = net_fine_copy
                        self.nerf_net_coarse = net_coarse_copy
                    raw_features_diff = torch.stack(raw_features_diff, -1)
                    raw_features_diff = torch.mean(raw_features_diff, -1)
                if not self.config['experiment']['merge_instance']:
                    difference_maps.append(raw_features_diff)
            difference_maps = torch.stack(difference_maps, -1)
            sem_fine = raw2outputs(raw_fine, z_vals, rays_d, raw_noise_std, self.white_bkgd,endpoint_feat=self.endpoint_feat,raw_semantic=difference_maps)
            sem = sem_fine
        return sem

    def render_propagate(self, tree, pts_label, subset='test', label_map=None, agg=None):
        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1)
        # render and save training-set images
        self.training = False  # enable testing mode before rendering results, need to set back during training!
        self.nerf_net_coarse.eval()
        self.nerf_net_fine.eval()
        savedir = os.path.join(self.config["experiment"]["save_dir"], subset + "_render")
        os.makedirs(savedir, exist_ok=True)
        if subset == 'train':
            print(' {} train images'.format(self.num_train))
        elif subset == 'test':
            print(' {} test images'.format(self.num_test))
        with torch.no_grad():
            if subset == 'train':
                rays = self.rays_vis
            elif subset == 'test':
                rays = self.rays_test
        for i, c2w in enumerate(tqdm(rays)):
            torch.cuda.empty_cache()
            output = self.render_rays_propagate(rays[i], tree, pts_label)
            # normalize in 2D
            if self.config['experiment']['normalize_2d']:
                output = output / (torch.max(output, dim=0)[0] + 1e-7)
            if self.config['experiment']['test_instance']:
                test_semantic = self.test_instance_scaled[i]
            else:
                test_semantic = self.test_semantic_scaled[i]
            if self.cpu:
                if self.config['experiment']['test_instance']:
                    color_map = self.instance_colour_map.numpy()
                else:
                    color_map = self.valid_colour_map.numpy()
            else:
                if self.config['experiment']['test_instance']:
                    color_map = self.instance_colour_map.cpu().numpy()
                else:
                    color_map = self.valid_colour_map.cpu().numpy()
            if agg is not None:
                output_agg = output.reshape((-1, output.shape[-1])).cuda()
                with torch.no_grad():
                    output_agg = agg.agg_net(output_agg)
                label_agg = logits_2_label(output_agg).cpu().numpy().reshape((self.H_scaled, self.W_scaled))
                if label_map is not None:
                    label_agg = label_map[label_agg].astype(np.uint8)
                cv2.imwrite(os.path.join(savedir, str(i) + '_label.png'), label_agg)
                cv2.imwrite(os.path.join(savedir, str(i) + '_vis_label.png'), color_map[label_agg])
                error_map_agg = np.where(np.abs(label_agg - test_semantic) > 0, 255, 0)
                cv2.imwrite(os.path.join(savedir, str(i) + '_error_map.png'), error_map_agg)

                if not self.config['experiment']['merge_instance']:
                    if self.cpu:
                        label = logits_2_label(output).numpy().reshape((self.H_scaled, self.W_scaled))
                    else:
                        label = logits_2_label(output).cpu().numpy().reshape((self.H_scaled, self.W_scaled))
                    if label_map is not None:
                        label = label_map[label].astype(np.uint8)
                    cv2.imwrite(os.path.join(savedir, str(i) + '_label_wo.png'), label)
                    cv2.imwrite(os.path.join(savedir, str(i) + '_vis_label_wo.png'), color_map[label])
                    error_map = np.where(np.abs(label - test_semantic) > 0, 255, 0)
                    cv2.imwrite(os.path.join(savedir, str(i) + '_error_map_wo.png'), error_map)
            else:
                if self.cpu:
                    label = logits_2_label(output).numpy().reshape((self.H_scaled, self.W_scaled))
                else:
                    label = logits_2_label(output).cpu().numpy().reshape((self.H_scaled, self.W_scaled))
                if label_map is not None:
                    label = label_map[label].astype(np.uint8)
                cv2.imwrite(os.path.join(savedir, str(i) + '_label.png'), label)
                cv2.imwrite(os.path.join(savedir, str(i) + '_gt_label.png'), test_semantic)
                cv2.imwrite(os.path.join(savedir, str(i) + '_vis_label.png'), color_map[label])
                error_map = np.where(np.abs(label - test_semantic) > 0, 255, 0)
                cv2.imwrite(os.path.join(savedir, str(i) + '_error_map.png'), error_map)
        return

    def nocs_transform(self, pts):
        '''
        pts: [num_pts, 3]
        '''
        pts_flat = pts.reshape((-1, 3))
        pts_origin = (self.to_origin_transform[:3, :3] @ pts_flat.T + self.to_origin_transform[:3, 3:]).T
        pts_nocs = pts_origin / self.extents[np.newaxis, ...]
        pts_nocs = pts_nocs.reshape(pts.shape)
        return pts_nocs

    def volumetric_rendering(self, ray_batch, get_sampled_pts=False):
        if ray_batch.shape == torch.Size([11]):
            ray_batch = ray_batch.unsqueeze(0)
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals) # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self.N_samples])
        
        if self.perturb > 0. and self.training:  # perturb sampling depths (z_vals)
            if self.training is True:  # only add perturbation during training intead of testing
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).cuda()

                z_vals = lower + (upper - lower) * t_rand

        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts_sampled = pts_coarse_sampled

        raw_noise_std = self.raw_noise_std if self.training else 0
        raw_coarse = run_network(pts_coarse_sampled, viewdirs, self.nerf_net_coarse,
                                 self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)
        if self.config['experiment']['render_nocs']:
            xyz_coarse = self.nocs_transform(pts_coarse_sampled)
        else:
            xyz_coarse = None
        rgb_coarse, disp_coarse, acc_coarse, weights_coarse, depth_coarse, feat_map_coarse, xyz_map_coarse = \
            raw2outputs(raw_coarse, z_vals, rays_d, raw_noise_std, self.white_bkgd, endpoint_feat=False, pts_sampled=xyz_coarse)
        torch.cuda.empty_cache()

        if self.N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self.N_importance,
                                   det=(self.perturb == 0.) or (not self.training))
            z_samples = z_samples.detach()
            # detach so that grad doesn't propagate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
            pts_sampled = pts_fine_sampled

            raw_fine = run_network(pts_fine_sampled, viewdirs, lambda x: self.nerf_net_fine(x, self.endpoint_feat),
                        self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)

            if self.config['experiment']['render_nocs']:
                xyz_fine = self.nocs_transform(pts_fine_sampled)
            else:
                xyz_fine = None
            rgb_fine, disp_fine, acc_fine, weights_fine, depth_fine, feat_map_fine, xyz_map_fine = \
                raw2outputs(raw_fine, z_vals, rays_d, raw_noise_std, self.white_bkgd, endpoint_feat = self.endpoint_feat, pts_sampled=xyz_fine)

        ret = {}
        ret['raw_coarse'] = raw_coarse
        ret['rgb_coarse'] = rgb_coarse
        ret['disp_coarse'] = disp_coarse
        ret['acc_coarse'] = acc_coarse
        ret['depth_coarse'] = depth_coarse
        if self.config['experiment']['render_nocs']:
            ret['xyz_coarse'] = xyz_map_coarse

        if self.N_importance > 0:
            ret['rgb_fine'] = rgb_fine
            ret['disp_fine'] = disp_fine
            ret['acc_fine'] = acc_fine
            ret['depth_fine'] = depth_fine
            if self.config['experiment']['render_nocs']:
                ret['xyz_fine'] = xyz_map_fine
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            ret['raw_fine'] = raw_fine  # model's raw, unprocessed predictions.
            if self.endpoint_feat:
                ret['feat_map_fine'] = feat_map_fine
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
        if get_sampled_pts:
            return ret, pts_sampled, z_vals
        return ret

    def create_nerf(self):
        """Instantiate NeRF's MLP model.
        """
        nerf_model = NeRF
        embed_fn, input_ch = get_embedder(self.config["render"]["multires"], self.config["render"]["i_embed"], scalar_factor=10)

        input_ch_views = 0
        embeddirs_fn = None
        if self.config["render"]["use_viewdirs"]:
            embeddirs_fn, input_ch_views = get_embedder(self.config["render"]["multires_views"],
                                                        self.config["render"]["i_embed"],
                                                        scalar_factor=1)
        output_ch = 5 if self.N_importance > 0 else 4
        skips = [4]
        model = nerf_model(D=self.config["model"]["netdepth"], W=self.config["model"]["netwidth"],
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=self.config["render"]["use_viewdirs"]).cuda()

        for param in model.parameters():
            param.requires_grad = True
        grad_vars = list(model.parameters())
        if self.config["experiment"]["fix_density"]:
            for param in model.alpha_linear.parameters():
                param.requires_grad = False
            grad_vars = list(set(grad_vars)-set(model.alpha_linear.parameters()))

        model_fine = None
        if self.N_importance > 0:
            model_fine = nerf_model(D=self.config["model"]["netdepth_fine"], W=self.config["model"]["netwidth_fine"],
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=self.config["render"]["use_viewdirs"]).cuda()
            for param in model_fine.parameters():
                param.requires_grad = True
            grad_vars += list(model_fine.parameters())

            if self.config["experiment"]["fix_density"]:
                for param in model_fine.alpha_linear.parameters():
                    param.requires_grad = False
                grad_vars = list(set(grad_vars)-set(model_fine.alpha_linear.parameters()))
        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate)

        start = 0
        # Load checkpoints
        if self.config["experiment"]["ckpt_path"] is not None:
            ckpt_path = self.config["experiment"]["ckpt_path"]
            print('Found ckpt', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cuda:0')
            if self.config["experiment"]["resume"]:
                start = ckpt['global_step']
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_coarse_state_dict'], strict=False)
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'], strict=False)

        self.nerf_net_coarse = model
        self.nerf_net_fine = model_fine
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.optimizer = optimizer

        return start

    # optimization step
    def step(
        self,
        global_step,
    ):
        # Misc
        img2mse = lambda x, y: torch.mean((x - y) ** 2)
        mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())

        gt_feature_sim = self.config["experiment"]["gt_feature_sim"]
        sample_semantic = self.sample_semantic
        if self.config['experiment']['sample_alter']:
            if global_step % self.config['experiment']['finetune_label_step']:
                gt_feature_sim = True
                sample_semantic = False
            else:
                gt_feature_sim = False
                sample_semantic = True

        # sample rays to query and optimise
        from_unmasked = sample_semantic & self.config['experiment']['from_unmasked'] or self.config['experiment']['from_unmasked_only']
        sampled_data = self.sample_data(global_step, self.rays, self.H, self.W, no_batching=self.no_batching, mode="train", test_semantic=sample_semantic, fixed=self.config['experiment']['sample_fixed'], sample_pos=(self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]), from_unmasked=from_unmasked)
        if (self.load_dino or self.load_lseg) and sample_semantic:
            if not (self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]):
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available, sampled_descriptors = sampled_data
            elif self.no_batching or from_unmasked:
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available, sampled_descriptors, index_batch, index_hw = sampled_data
            else:
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available, sampled_descriptors, index_hw = sampled_data
        elif self.load_dino or self.load_lseg:
            if not (self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]):
                sampled_rays, sampled_gt_rgb, sampled_descriptors = sampled_data
            elif self.no_batching or from_unmasked:
                sampled_rays, sampled_gt_rgb, sampled_descriptors, index_batch, index_hw = sampled_data
            else:
                sampled_rays, sampled_gt_rgb, sampled_descriptors, index_hw = sampled_data
        elif sample_semantic:
            if not (self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]):
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available = sampled_data
            elif self.no_batching or from_unmasked:
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available, index_batch, index_hw = sampled_data
            else:
                sampled_rays, sampled_gt_rgb, sampled_gt_semantic, sematic_available, index_hw = sampled_data
        else:
            if not (self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]):
                sampled_rays, sampled_gt_rgb = sampled_data
            elif self.no_batching or from_unmasked:
                sampled_rays, sampled_gt_rgb, index_batch, index_hw = sampled_data
            else:
                sampled_rays, sampled_gt_rgb, index_hw = sampled_data

        if self.config['experiment']['visualize_sample']:
            if self.no_batching:
                return index_batch, index_hw
            else:
                return index_hw
        output_dict, sampled_pts, z_vals = self.render_rays(sampled_rays, get_sampled_pts=True)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        if self.config['experiment']['render_nocs']:
            nocs_coarse = output_dict["xyz_coarse"]

        if self.N_importance > 0:
            rgb_fine = output_dict["rgb_fine"]
            depth_fine = output_dict["depth_fine"]
            if self.config['experiment']['render_nocs']:
                nocs_fine = output_dict["xyz_fine"]

        self.optimizer.zero_grad()

        img_loss_coarse = img2mse(rgb_coarse, sampled_gt_rgb)

        with torch.no_grad():
            psnr_coarse = mse2psnr(img_loss_coarse)

        if self.N_importance > 0:
            img_loss_fine = img2mse(rgb_fine, sampled_gt_rgb)
            with torch.no_grad():
                psnr_fine = mse2psnr(img_loss_fine)
        else:
            img_loss_fine = torch.tensor(0)
            psnr_fine = torch.tensor(0)

        total_img_loss = img_loss_coarse + img_loss_fine

        wgt_img_loss = float(self.config["train"]["wgt_img"])
        wgt_contrastive_loss = float(self.config["experiment"]["wgt_contrastive"])
        wgt_gradient_loss = float(self.config["experiment"]["wgt_gradient"])

        total_loss = total_img_loss*wgt_img_loss

        mi_contrastive_loss = torch.tensor(0, dtype=torch.float64, device='cuda')
        gradient_loss = torch.tensor(0, dtype=torch.float64, device='cuda')
        gradient_norm = torch.tensor(0, dtype=torch.float64, device='cuda')

        # add contrastive loss
        if (self.config["experiment"]["contrastive_2d"] or self.config["experiment"]["contrastive_3d"] or self.config["experiment"]["from_label"]) and global_step >= self.config["experiment"]["contrastive_starting_step"] and global_step % self.config["experiment"]["contrastive_step"] == 0:
            torch.cuda.empty_cache()
            if self.config['experiment']['render_nocs']:
                pts = nocs_fine.detach()
            elif self.config['experiment']['spatial'] or self.config['experiment']['spatial_embedding'] or self.config["experiment"]["contrastive_3d"]:
                pts = np.zeros((0, 3))
                for index, depth in enumerate(depth_fine):
                    depths_flat = np.zeros(self.n_rays, dtype=np.float64)
                    depths_flat[index] = depth.detach().cpu().numpy().astype(np.float64)
                    if self.no_batching or from_unmasked:
                        depths = np.zeros((self.num_train, self.H*self.W), dtype=np.float64)
                        depths[index_batch, index_hw] = depths_flat
                    else:
                        depths = np.zeros((self.num_train*self.H*self.W), dtype=np.float64)
                        depths[index_hw] = depths_flat
                    depths = depths.reshape((self.num_train, self.H, self.W))
                    depth_index = np.nonzero(depths)[0]
                    sub_pts = back_project(depths[depth_index, ...].squeeze(), self.K, self.train_Ts[depth_index].squeeze().numpy())
                    pts = np.concatenate((pts, sub_pts), axis=0)
                pts = torch.tensor(pts, dtype=torch.float32).cuda()

            jacobians = []
            if self.config["experiment"]["contrastive_2d"]:
                model_features = rgb_fine
            elif self.config["experiment"]["contrastive_3d"]:
                raw = run_network(pts.unsqueeze(1), torch.zeros_like(pts, dtype=torch.float32),
                                  lambda x: self.nerf_net_fine(x, self.endpoint_feat),
                                  self.embed_fn, self.embeddirs_fn, netchunk=self.netchunk)
                model_features = torch.sigmoid(raw[..., :3])
            for feature in model_features:
                self.clean_grad()
                if self.config["experiment"]["random_channel"]:
                    rand_channel = np.random.randint(0, 3)
                    feature_model = feature[rand_channel]
                else:
                    feature_model = feature.mean()
                jacobian = self.get_jacobian(outputs=feature_model)[0].view(-1)
                jacobians.append(jacobian)
            jacobians = torch.stack(jacobians, 0)

            gradient_norm = torch.mean(torch.linalg.norm(jacobians, dim=-1))

            if self.config["experiment"]["nce_dot_product"] or self.config["experiment"]["gradient_norm_loss"]:
                gradient_loss += torch.mean((torch.linalg.norm(jacobians, dim=-1) - 1.0) ** 2)

            if gt_feature_sim:
                sampled_features = sampled_descriptors.detach()
                sampled_features = sampled_features / (torch.linalg.norm(sampled_features, dim=-1).unsqueeze(-1) + 1e-7)
            elif sample_semantic:
                sampled_features = sampled_gt_semantic.detach()

            if self.config['experiment']['spatial_embedding']:
                pts = self.embed_fn(pts)
            if self.config['experiment']['render_nocs']:
                sampled_nocs = pts
            elif gt_feature_sim and self.config['experiment']['spatial_embedding'] or self.config['experiment']['spatial']:
                pts = pts / (torch.linalg.norm(pts, dim=-1).unsqueeze(-1) + 1e-7)
                sampled_features = torch.hstack((sampled_features, pts))

            if gt_feature_sim and self.config['experiment']['concat_color']:
                color = sampled_gt_rgb / (torch.linalg.norm(sampled_gt_rgb, dim=-1).unsqueeze(-1) + 1e-7)
                sampled_features = torch.hstack((sampled_features, color))

            anchor_num = 0
            if gt_feature_sim:
                sampled_features = sampled_features / (torch.linalg.norm(sampled_features, dim=-1).unsqueeze(-1) + 1e-7)

            for index, sample_feature in enumerate(sampled_features):
                torch.cuda.empty_cache()
                if sample_semantic:
                    sim_list = torch.tensor(sampled_features==sample_feature)
                else:
                    sim_list = sampled_features @ sample_feature
                if self.config['experiment']['render_nocs']:
                    sample_nocs = sampled_nocs[index, :]
                    nocs_sim_list = sampled_nocs @ sample_nocs
                    pos_condition = torch.tensor((sim_list > self.pos_threshold) & (nocs_sim_list > self.nocs_pos_threshold))
                elif sample_semantic:
                    pos_condition = sim_list
                else:
                    pos_condition = torch.tensor(sim_list > self.pos_threshold)
                if self.config["experiment"]["adaptive_threshold"] and gt_feature_sim and not self.config["experiment"]["pos_order"]:
                    if pos_condition.sum() < int(self.config["experiment"]["pos_ratio_lower"] * self.n_rays) and self.pos_threshold > self.config["experiment"]["pos_threshold_lower"]:
                        self.pos_threshold -= 0.001
                    elif pos_condition.sum() > int(self.config["experiment"]["pos_ratio_upper"] * self.n_rays) and self.pos_threshold < self.config["experiment"]["pos_threshold_upper"]:
                        self.pos_threshold += 0.001
                pos_sim = torch.tensor(0, dtype=torch.float64, device='cuda')
                if self.config["experiment"]["pos_order"]:
                    if self.config["experiment"]["nce_dot_product"]:
                        if self.config['experiment']['contrastive_abs']:
                            pos_sim += torch.exp(abs(torch.dot(jacobians[np.argmax(sim_list)], jacobians[index])))
                        else:
                            pos_sim += torch.exp(torch.dot(jacobians[np.argmax(sim_list)], jacobians[index]))
                    else:
                        if self.config['experiment']['contrastive_abs']:
                            pos_sim += torch.exp(abs(torch.dot(jacobians[np.argmax(sim_list)], jacobians[index]) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(jacobians[np.argmax(sim_list)]) + 1e-7)))
                        else:
                            pos_sim += torch.exp(torch.dot(jacobians[np.argmax(sim_list)], jacobians[index]) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(jacobians[np.argmax(sim_list)]) + 1e-7))
                else:
                    for pos_sample in jacobians[pos_condition]:
                        if self.config["experiment"]["nce_dot_product"]:
                            if self.config['experiment']['contrastive_abs']:
                                pos_sim += torch.exp(abs(torch.dot(jacobians[index], pos_sample)))
                            else:
                                pos_sim += torch.exp(torch.dot(jacobians[index], pos_sample))
                        else:
                            if self.config['experiment']['contrastive_abs']:
                                pos_sim += torch.exp(abs(torch.dot(jacobians[index], pos_sample) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(pos_sample) + 1e-7)))
                            else:
                                pos_sim += torch.exp(torch.dot(jacobians[index], pos_sample) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(pos_sample) + 1e-7))
                neg_sim = torch.tensor(0, dtype=torch.float64, device='cuda')
                for neg_sample in jacobians[~pos_condition]:
                    if self.config["experiment"]["nce_dot_product"]:
                        if self.config['experiment']['contrastive_abs']:
                            neg_sim += torch.exp(abs(torch.dot(jacobians[index], neg_sample)))
                        else:
                            neg_sim += torch.exp(torch.dot(jacobians[index], neg_sample))
                    else:
                        if self.config['experiment']['contrastive_abs']:
                            neg_sim += torch.exp(abs(torch.dot(jacobians[index], neg_sample) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(neg_sample) + 1e-7)))
                        else:
                            neg_sim += torch.exp(torch.dot(jacobians[index], neg_sample) / (torch.linalg.norm(jacobians[index]) * torch.linalg.norm(neg_sample) + 1e-7))
                if pos_sim > 0 and neg_sim > 0:
                    mi_contrastive_loss -= torch.log(pos_sim / (pos_sim + neg_sim) + 1e-7)
                    anchor_num += 1
            if anchor_num > 0:
                mi_contrastive_loss /= anchor_num

            total_loss += wgt_contrastive_loss * mi_contrastive_loss + wgt_gradient_loss * gradient_loss

        total_loss.backward()
        self.optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = self.lrate_decay
        new_lrate = self.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # tensorboard-logging
        # visualize loss curves
        if global_step % float(self.config["logging"]["step_log_tfb"])==0:
            self.tfb_viz.vis_scalars(global_step,
                                    [img_loss_coarse, img_loss_fine, total_img_loss,
                                    mi_contrastive_loss, wgt_contrastive_loss * mi_contrastive_loss,
                                    gradient_loss, wgt_gradient_loss * gradient_loss,
                                    total_loss, gradient_norm],
                                    ['Train/Loss/img_loss_coarse', 'Train/Loss/img_loss_fine', 'Train/Loss/total_img_loss',
                                     'Train/Loss/contrastive_loss', 'Train/Loss/weighted_total_contrastive_loss',
                                     'Train/Loss/gradient_loss', 'Train/Loss/weighted_total_gradient_loss',
                                    'Train/Loss/total_loss', 'Train/Gradient/gradient_norm'])

            # add raw transparancy value into tfb histogram
            trans_coarse = output_dict["raw_coarse"][..., 3]   
            self.tfb_viz.vis_histogram(global_step, trans_coarse, 'trans_coarse')     
            if self.N_importance>0:
                trans_fine = output_dict['raw_fine'][..., 3]   
                self.tfb_viz.vis_histogram(global_step, trans_fine, 'trans_fine')

            self.tfb_viz.vis_scalars(global_step,
                        [psnr_coarse, psnr_fine],
                        ['Train/Metric/psnr_coarse', 'Train/Metric/psnr_fine'])

        # Rest is logging, saving ckpts regularly
        if global_step % float(self.config["logging"]["step_save_ckpt"])==0:
            ckpt_dir = os.path.join(self.save_dir, "checkpoints")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            ckpt_file = os.path.join(ckpt_dir, '{:06d}.ckpt'.format(global_step))
            torch.save({
                'global_step': global_step,
                'network_coarse_state_dict': self.nerf_net_coarse.state_dict(),
                'network_fine_state_dict': self.nerf_net_fine.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_file)
            print('Saved checkpoints at', ckpt_file)

        # render and save training-set images
        if global_step % self.config["logging"]["step_vis_train"]==0 and global_step > 0:
            self.training = False  # enable testing mode before rendering results, need to set back during training!
            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()
            trainsavedir = os.path.join(self.config["experiment"]["save_dir"], "train_render", 'step_{:06d}'.format(global_step))
            os.makedirs(trainsavedir, exist_ok=True)
            print(' {} train images'.format(self.num_train))
            with torch.no_grad():
                rgbs, disps, deps, vis_deps = self.render_path(self.rays_vis, save_dir=trainsavedir)
                #  numpy array of shape [B,H,W,C] or [B,H,W]
            print('Saved training set')

            self.training = True  # set training flag back after rendering images
            self.nerf_net_coarse.train()
            self.nerf_net_fine.train()

            with torch.no_grad():
                batch_train_img_mse = img2mse(torch.from_numpy(rgbs), self.train_image_scaled.cpu())
                batch_train_img_psnr = mse2psnr(batch_train_img_mse)
                self.tfb_viz.vis_scalars(global_step, [batch_train_img_psnr, batch_train_img_mse], ['Train/Metric/batch_PSNR', 'Train/Metric/batch_MSE'])

            imageio.mimwrite(os.path.join(trainsavedir, 'rgb.mp4'), to8b_np(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(trainsavedir, 'dep.mp4'), vis_deps, fps=30, quality=8)
            imageio.mimwrite(os.path.join(trainsavedir, 'disps.mp4'), to8b_np(disps / np.max(disps)), fps=30, quality=8)

            # add rendered image into tf-board
            self.tfb_viz.tb_writer.add_image('Train/rgb', rgbs, global_step, dataformats='NHWC')
            self.tfb_viz.tb_writer.add_image('Train/depth', vis_deps, global_step, dataformats='NHWC')
            self.tfb_viz.tb_writer.add_image('Train/disps', np.expand_dims(disps,-1), global_step, dataformats='NHWC')

        # render and save test images, corresponding videos
        if global_step % self.config["logging"]["step_val"]==0 and global_step > 0:
            self.training = False  # enable testing mode before rendering results, need to set back during training!
            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()
            testsavedir = os.path.join(self.config["experiment"]["save_dir"], "test_render", 'step_{:06d}'.format(global_step))
            os.makedirs(testsavedir, exist_ok=True)
            print(' {} test images'.format(self.num_test))
            with torch.no_grad():
                rgbs, disps, deps, vis_deps = self.render_path(self.rays_test, save_dir=testsavedir)
            print('Saved test set')

            self.training = True  # set training flag back after rendering images
            self.nerf_net_coarse.train()
            self.nerf_net_fine.train()

            with torch.no_grad():
                batch_test_img_mse = img2mse(torch.from_numpy(rgbs), self.test_image_scaled.cpu())
                batch_test_img_psnr = mse2psnr(batch_test_img_mse)
                
                self.tfb_viz.vis_scalars(global_step, [batch_test_img_psnr, batch_test_img_mse], ['Test/Metric/batch_PSNR', 'Test/Metric/batch_MSE'])

            imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b_np(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'dep.mp4'), vis_deps, fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'disps.mp4'), to8b_np(disps / np.max(disps)), fps=30, quality=8)

            # add rendered image into tf-board
            self.tfb_viz.tb_writer.add_image('Test/rgb', rgbs, global_step, dataformats='NHWC')
            self.tfb_viz.tb_writer.add_image('Test/depth', vis_deps, global_step, dataformats='NHWC')
            self.tfb_viz.tb_writer.add_image('Test/disps', np.expand_dims(disps,-1), global_step, dataformats='NHWC')

        if global_step%self.config["logging"]["step_log_print"]==0:
            tqdm.write(f"[TRAIN] Iter: {global_step} "
                       f"Loss: {total_loss.item()} "
                       f"rgb_total_loss: {total_img_loss.item()}, rgb_coarse: {img_loss_coarse.item()}, rgb_fine: {img_loss_fine.item()}, "
                       f"PSNR_coarse: {psnr_coarse.item()}, PSNR_fine: {psnr_fine.item()}"
                       f"contrastive: {mi_contrastive_loss.item()}, weighted_contrastive: {mi_contrastive_loss.item()*wgt_contrastive_loss} "
                       f"gradient: {gradient_loss.item()}, weighted_gradient: {wgt_gradient_loss * gradient_loss.item()}")

    def render_only(self, subset='test', save_idx=0, save_img=True):
        # render and save training-set images
        self.training = False  # enable testing mode before rendering results, need to set back during training!
        self.nerf_net_coarse.eval()
        self.nerf_net_fine.eval()
        savedir = os.path.join(self.config["experiment"]["save_dir"], subset + "_render")
        os.makedirs(os.path.join(savedir, str(save_idx)), exist_ok=True)
        if subset == 'train':
            print(' {} train images'.format(self.num_train))
        elif subset == 'test':
            print(' {} test images'.format(self.num_test))

        with torch.no_grad():
            if subset == 'train':
                rays = self.rays_vis
            elif subset == 'test':
                rays = self.rays_test

            rgbs, disps, deps, vis_deps = self.render_path(rays, save_dir=os.path.join(savedir, str(save_idx)), save_img=save_img)
        print('Saved subset')
        return to8b_np(rgbs)

    def render_path(self, rays, save_dir=None, save_img=True, idx=None):
        rgbs = []
        disps = []
        deps = []
        vis_deps = []
        nocs_maps = []

        t = time.time()
        for i, c2w in enumerate(tqdm(rays)):
            print(i, time.time() - t)
            t = time.time()
            output_dict = self.render_rays(rays[i], test=True)
            rgb_coarse = output_dict["rgb_coarse"]
            disp_coarse = output_dict["disp_coarse"]
            depth_coarse = output_dict["depth_coarse"]
            rgb = rgb_coarse
            depth = depth_coarse
            disp = disp_coarse

            if self.config['experiment']['render_nocs']:
                nocs_map = output_dict['xyz_coarse']

            if self.N_importance > 0:
                rgb_fine = output_dict["rgb_fine"]
                depth_fine = output_dict["depth_fine"]
                disp_fine = output_dict["disp_fine"]

                rgb = rgb_fine
                depth = depth_fine
                disp = disp_fine

                if self.config['experiment']['render_nocs']:
                    nocs_map = output_dict['xyz_fine']

                torch.cuda.empty_cache()
        
            rgb = rgb.cpu().numpy().reshape((self.H_scaled, self.W_scaled, 3))
            depth = depth.cpu().numpy().reshape((self.H_scaled, self.W_scaled))
            disp = disp.cpu().numpy().reshape((self.H_scaled, self.W_scaled))

            rgbs.append(rgb)
            disps.append(disp)
            deps.append(depth)  # save depth in mm
            vis_deps.append(depth2rgb(depth, min_value=self.near, max_value=self.far))

            if self.config['experiment']['render_nocs']:
                nocs_map = nocs_map.cpu().numpy().reshape((self.H_scaled, self.W_scaled, 3))
                nocs_maps.append(nocs_map)

            if save_dir is not None and save_img:
                os.makedirs(save_dir, exist_ok=True)
                rgb8 = (rgbs[-1]*255).astype(np.uint8)
                disp = disps[-1].astype(np.uint16)
                dep_mm = (deps[-1]*1000).astype(np.uint16)
                vis_dep = vis_deps[-1]

                rgb_filename = os.path.join(save_dir, 'rgb_{:03d}.png'.format(i))
                disp_filename = os.path.join(save_dir, 'disp_{:03d}.png'.format(i))

                depth_filename = os.path.join(save_dir, 'depth_{:03d}.png'.format(i))
                vis_depth_filename = os.path.join(save_dir, 'vis_depth_{:03d}.png'.format(i))

                imageio.imwrite(rgb_filename, rgb8)
                imageio.imwrite(disp_filename, disp, format="png", prefer_uint8=False)

                imageio.imwrite(depth_filename, dep_mm, format="png", prefer_uint8=False)
                imageio.imwrite(vis_depth_filename, vis_dep)

                if self.config['experiment']['render_nocs']:
                    nocs8 = to8b_np(nocs_maps[-1])
                    nocs_filename = os.path.join(save_dir, 'nocs_{:03d}.png'.format(i))
                    imageio.imwrite(nocs_filename, nocs8)

            if idx is not None and i == idx:
                break

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)
        deps = np.stack(deps, 0)
        vis_deps = np.stack(vis_deps, 0)
        return rgbs, disps, deps, vis_deps

    def get_given_labels_jacobians(self):
        print("process train view label")
        print("mask_ids:", self.mask_ids)
        if self.config['experiment']['test_instance']:
            sem_train = self.train_instance[self.mask_ids].cpu().numpy()
        else:
            sem_train = self.train_semantic_scaled[self.mask_ids]
        mask = np.array(sem_train >= 0)
        sem = sem_train.reshape(sem_train.shape[0], -1)
        points_label = sem_train[mask]
        points_label = np.unique(points_label).astype(np.int8)
        logits_2_label = lambda x: np.argmax(softmax(x, axis=-1), axis=-1)

        if self.config['experiment']['adaptive_selection']:
            agg_trainer = None
            ignore_label = []
            ignore_label.append(-1)
            existing_class_mask = np.ones(self.num_valid_semantic_class, dtype=np.bool)
            for label in range(self.num_valid_semantic_class):
                if label not in points_label:
                    existing_class_mask[label] = False
                    ignore_label.append(label)
            with torch.no_grad():
                rgb, _, _, _ = self.render_path(self.rays_vis[self.mask_ids], None, save_img=False)
            jacobians_selection = []
            responses_selection = []
            mask_labels = np.stack([np.array(sem == label).squeeze() for label in points_label], 0)
            max_miou = 0
            i = 0
            while i < self.config['experiment']['num_iters']:
                logits = []
                jacobians = []
                jacobians_ids = []
                responses = []
                # random sample num_comb combinations each time, get gradient, self-perturb
                for idx, label in enumerate(tqdm(points_label)):
                    response = []
                    raw_response = []
                    mask_label = mask_labels[idx, ...]
                    rand_selection = np.random.randint(0, mask_label.sum(), (1, self.config['experiment']['num_comb']))
                    grads = self.render_gradient_rays(self.rays_vis[self.mask_ids].squeeze()[mask_label, ...][rand_selection, ...].squeeze())
                    # perturb along the gradient and get response
                    for grad in grads:
                        net_fine_copy = copy.deepcopy(self.nerf_net_fine)
                        net_coarse_copy = copy.deepcopy(self.nerf_net_coarse)
                        self.perturb_one_direction(grad, t=self.config['experiment']['perturb_t'])
                        with torch.no_grad():
                            rgb_p, _, _, _ = self.render_path(self.rays_vis[self.mask_ids], None, save_img=False)
                        if self.config['experiment']['no_abs']:
                            difference = np.mean((rgb_p - rgb), -1)
                        else:
                            difference = np.mean(abs(rgb_p - rgb), -1)
                        if i > 0:
                            conditional_response = [res[..., idx] for res in responses_selection]
                            conditional_response.append(difference)
                            conditional_response = np.stack(conditional_response, -1)
                            difference = np.mean(conditional_response, -1)
                        raw_response.append(difference)
                        difference[difference > 0.8] = 0.8
                        difference = cv2.GaussianBlur(difference, (3, 3), 0)
                        difference = difference / (np.max(difference) + 1e-7)
                        # exclude the case when this label doesn't appear in this view
                        if np.std(difference) < 0.05 and np.mean(difference) < 0.01:
                            difference = np.zeros_like(difference)
                        response.append(difference)
                        self.nerf_net_fine = net_fine_copy
                        self.nerf_net_coarse = net_coarse_copy
                    response = np.stack(response, 0)
                    raw_response = np.stack(raw_response, 0)
                    responses.append(raw_response)
                    logits.append(response)
                    jacobians.append(grads)
                    jacobians_ids.append(np.argwhere(mask_label > 0)[rand_selection, ...].squeeze())
                responses = np.stack(responses, -1)
                logits = np.stack(logits, -1)
                labels = points_label[logits_2_label(logits)]
                jacobians = torch.stack(jacobians, 0)
                mious = []
                for label in labels:
                    # todo: only evaluate the given regions under sparse setting, i.e. exclude void classes
                    miou = []
                    for view_i in range(self.mask_ids.sum()):
                        miou_i, miou_validclass, total_accuracy, class_average_accuracy, ious, class_average_accuracy_validclass = calculate_segmentation_metrics(true_labels=sem_train[view_i, ...], predicted_labels=label[view_i, ...], number_classes=self.num_valid_semantic_class, ignore_label=ignore_label, class_mask=existing_class_mask)
                        miou.append(miou_i)
                    miou = np.mean(miou)
                    mious.append(miou)
                mious = np.stack(mious)
                if np.max(mious) > max_miou:
                    jacobians_selection.append(jacobians[:, np.argmax(mious), ...].squeeze())
                    responses_selection.append(responses[np.argmax(mious), ...])
                    i += 1

            jacobians_selection = torch.stack(jacobians_selection, dim=1)

            if self.config['experiment']['train_agg']:
                if self.config['experiment']['propagate_3d']:
                    if self.config['experiment']['merge_instance']:
                        # original responses for 3D
                        with torch.no_grad():
                            responses_selection = self.render_rays_propagate(self.rays_vis[self.mask_ids].reshape(-1, self.rays_vis.shape[-1]), jacobians_selection, points_label)
                    else:
                        responses_selection = []
                        for i in tqdm(range(1, self.config['experiment']['num_iters']+1)):
                            torch.cuda.empty_cache()
                            with torch.no_grad():
                                output = self.render_rays_propagate(self.rays_vis[self.mask_ids].reshape(-1, self.rays_vis.shape[-1]), jacobians_selection[:, :i, ...], points_label)
                            responses_selection.append(output.reshape((self.mask_ids.sum(), self.H_scaled, self.W_scaled, -1)).cpu().numpy())
                        responses_selection = np.stack(responses_selection, 0)
                else:
                    responses_selection = np.stack(responses_selection, 0)
                label_remap = -np.ones_like(sem.reshape(-1), dtype=np.int8)
                for i in range(len(points_label)):
                    label_remap[sem.reshape(-1) == points_label[i]] = i
                label_remap = torch.tensor(label_remap).long().cuda()
                if self.config['experiment']['merge_instance']:
                    agg_trainer = aggregator_trainer.AggregatorTrainer(len(points_label), lrate=1e-3, lrate_decay=500e3, input_class=responses_selection.shape[-1])
                else:
                    agg_trainer = aggregator_trainer.AggregatorTrainer(len(points_label), lrate=1e-3, lrate_decay=500e3)
                start = agg_trainer.create_model()
                agg_trainer.agg_net.to(torch.device('cuda'))
                N_iters = 200000
                ##########################
                print('Begin')
                x = []
                losses = []
                #####  Training loop  #####
                for i in trange(start, N_iters):
                    x.append(i)
                    if self.config['experiment']['merge_instance']:
                        sampled_response = responses_selection.cuda()
                    else:
                        sampled_idx = np.random.choice(len(responses_selection), 1)
                        sampled_response = torch.tensor(responses_selection[sampled_idx, ...]).reshape((-1, len(points_label))).cuda()
                    loss = agg_trainer.step(i, sampled_response, label_remap)
                    losses.append(loss)
                print('done, loss:', losses[-1])
            return sem_train, sem, mask, points_label, jacobians_selection, agg_trainer

        if self.config['experiment']['gradient_kmeans']:
            kmeans = KMeans(n_clusters=self.config['experiment']['n_clusters'], mode='cosine')
        jacobian_train = []
        for label in tqdm(points_label):
            mask_label = np.array(sem == label)
            grad = []
            for i in range(0, mask_label.sum(), 1024):
                jac = self.render_gradient_rays(self.rays_vis[self.mask_ids][mask_label, ...][i:i+1024, ...])
                grad.append(jac)
            grad = torch.cat(grad, 0)
            if self.config['experiment']['mean_gradient']:
                grad = torch.mean(grad, dim=0)
            elif self.config['experiment']['gradient_kmeans'] and self.config['experiment']['n_clusters'] < mask_label.sum():
                gs = grad.reshape((mask_label.sum(), -1)).float()
                kmeans.fit_predict(gs)
                centers = kmeans.centroids
                grad = torch.stack([gs[torch.argmax(torch.matmul(gs, ctr)/(torch.linalg.norm(gs, dim=-1)*torch.linalg.norm(ctr)+1e-8)), :] for ctr in centers], 0)
                grad = grad.reshape((-1, 3, 128))
                if self.config['experiment']['visualize_gradients']:
                    grad_ids = torch.stack([torch.argmax(torch.matmul(gs, ctr)/(torch.linalg.norm(gs, dim=-1)*torch.linalg.norm(ctr)+1e-8)) for ctr in centers], 0)
                    grad_ids = grad_ids.cpu().numpy()
                    # TODO: given multi view images
                    img = np.zeros_like(mask_label.squeeze(), dtype=np.uint8)
                    img_grad_ids = np.argwhere(mask_label.squeeze() > 0)[grad_ids]
                    img[img_grad_ids] = 255
                    img = img.reshape((self.H_scaled, self.W_scaled)).astype(np.uint8)
                    cv2.imwrite(os.path.join(self.save_dir, str(label)+'_grad.png'), img)

            jacobian_train.append(grad)
        if self.config['experiment']['mean_gradient']:
            jacobian_train = torch.stack(jacobian_train, 0)
        return sem_train, sem, mask, points_label, jacobian_train, None