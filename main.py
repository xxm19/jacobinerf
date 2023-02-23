import numpy as np
import torch
import yaml
import os
import argparse
from datasets.replica import replica_datasets
from datasets.scannet import scannet_datasets
from training import trainer
from utils.geometry_utils import back_project
from tqdm import tqdm, trange
import time
import cv2
import copy
from scipy.special import softmax
to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./configs/replica_room0_config.yaml", help='config file name.')
    parser.add_argument('--test_instance', action='store_true', help='whether use instance label from scannet')
    parser.add_argument('--training_mode', action='store_true', help='do not load test images')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--render_only', action='store_true')
    parser.add_argument('--no_batching', action='store_true', default=False)
    parser.add_argument('--dataset_type', type=str, default="replica", choices= ["replica", "scannet"], help='the dataset to be used,')
    parser.add_argument('--lrate', type=float, default=5e-4)
    parser.add_argument('--lrate_decay', type=int, default=250e3)
    parser.add_argument('--N_rays', type=int, default=1024)
    parser.add_argument('--N_importance', type=int, default=128)
    parser.add_argument('--N_samples', type=int, default=64)
    parser.add_argument('--test_viz_factor', type=int, default=1)
    parser.add_argument('--wgt_img', type=float, default=1)

    # load pre-extracted dino/lseg feature
    parser.add_argument('--load_dino', action='store_true')
    parser.add_argument('--load_lseg', action='store_true')
    parser.add_argument('--high_res_dino', action='store_true')
    parser.add_argument('--feature_dim', type=int, default=384)
    parser.add_argument('--load_on_cpu', action='store_true', default=False)

    # J-NeRF finetuning
    parser.add_argument('--contrastive_2d', action='store_true')
    parser.add_argument('--contrastive_abs', action='store_true')
    parser.add_argument('--contrastive_3d', action='store_true')
    parser.add_argument('--wgt_contrastive', type=float, default=0.1)
    parser.add_argument('--wgt_gradient', type=float, default=0.1)
    parser.add_argument('--contrastive_step', type=int, default=5000)
    parser.add_argument('--contrastive_starting_step', type=int, default=10000)
    parser.add_argument('--spatial', action='store_true')
    parser.add_argument('--spatial_embedding', action='store_true')
    parser.add_argument('--concat_color', action='store_true', help='concat color with dino/lseg feature')
    parser.add_argument('--sample_semantic', action='store_true')
    parser.add_argument('--nce_dot_product', action='store_true')
    parser.add_argument('--gradient_norm_loss', action='store_true')
    parser.add_argument('--gt_feature_sim', action='store_true')
    parser.add_argument('--pos_order', action='store_true')
    parser.add_argument('--neg_threshold', type=float, default=0.8)
    parser.add_argument('--pos_threshold', type=float, default=0.8)
    parser.add_argument('--nocs_pos_threshold', type=float, default=0.9)
    parser.add_argument('--adaptive_threshold', action='store_true')
    parser.add_argument('--pos_ratio_upper', type=float, default=0.15)
    parser.add_argument('--pos_threshold_upper', type=float, default=0.9)
    parser.add_argument('--pos_ratio_lower', type=float, default=0.05)
    parser.add_argument('--pos_threshold_lower', type=float, default=0.5)
    parser.add_argument('--render_nocs', action='store_true')

    # which subset of parameters to shape
    parser.add_argument('--rgb_layer', action='store_true')
    parser.add_argument('--rgb_pts_layer', action='store_true')
    parser.add_argument('--density_layer', action='store_true')
    parser.add_argument('--pts_layer', action='store_true')
    parser.add_argument('--all_para', action='store_true')

    # other J-NeRF training args, not important
    parser.add_argument('--from_label', action='store_true', help='oracle case, acquire semantic covariance supervision from gt semantic label')
    parser.add_argument('--sample_alter', action='store_true', help='sample alternatively from given dense label view and other training views')
    parser.add_argument('--finetune_label_step', type=int, default=2)
    parser.add_argument('--from_unmasked', action='store_true', default=False)
    parser.add_argument('--from_unmasked_only', action='store_true', default=False)
    parser.add_argument('--sample_fixed', action='store_true')
    parser.add_argument('--fix_density', action='store_true')

    # label propagation
    parser.add_argument('--propagate_3d', action='store_true')
    parser.add_argument('--propagate_2d', action='store_true')
    parser.add_argument('--normalize_2d', action='store_true')
    parser.add_argument('--no_abs', action='store_true')
    parser.add_argument('--random_channel', action='store_true')
    parser.add_argument('--train_agg', action='store_true')
    parser.add_argument('--mean_response', action='store_true')
    parser.add_argument('--mean_gradient', action='store_true')
    parser.add_argument('--gradient_kmeans', action='store_true')
    parser.add_argument('--merge_instance', action='store_true')
    parser.add_argument('--adaptive_selection', action='store_true')
    parser.add_argument('--num_comb', type=int, default=20)
    parser.add_argument('--visualize_gradients', action='store_true')
    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--perturb_t', type=float, default=0.1)
    parser.add_argument('--perturb_r', type=int, default=255)
    parser.add_argument('--perturb_g', type=int, default=255)
    parser.add_argument('--perturb_b', type=int, default=255)
    parser.add_argument('--t_iter', action='store_true')

    # sparse-views
    parser.add_argument("--sparse_views", action='store_true',
                        help='Use labels from a sparse set of frames')
    parser.add_argument("--sparse_ratio", type=float, default=0,
                        help='The portion of dropped labelling frames during training, which can be used along with all working modes.')    
    parser.add_argument("--label_map_ids", nargs='*', type=int, default=[],
                        help='In sparse view mode, use selected frame ids from sequences as supervision.')
    parser.add_argument("--random_sample", action='store_true', help='Whether to randomly/evenly sample frames from the sequence.')

    # sparse pixels
    parser.add_argument("--label_propagation", action='store_true',
                        help='Label propagation using partial seed regions.')
    parser.add_argument("--partial_perc", type=float, default=0,
                        help='0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation')
    parser.add_argument("--user_click", action='store_true', help='simulate user clicking')
    parser.add_argument("--num_click", type=int, default=3, help='number of pixel-labels provided for each class, num_click=1 is equivalent to partial_perc=0')

    # misc.
    parser.add_argument('--visualise_save',  action='store_true', help='whether to save the noisy labels into harddrive for later usage')
    parser.add_argument('--load_saved',  action='store_true', help='use trained noisy labels for training to ensure consistency betwwen experiments')
    parser.add_argument('--gpu', type=str, default="", help='GPU IDs.')

    # visualizations, not important
    parser.add_argument('--perturb_gradient', action='store_true')
    parser.add_argument('--visualize_sample', action='store_true')

    args = parser.parse_args()
    # Read YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if len(args.gpu)>0:
        config["experiment"]["gpu"] = args.gpu
    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
    config["experiment"].update(vars(args))
    config["train"].update(vars(args))
    config["render"].update(vars(args))

    # Cast intrinsics to right types
    nerf_trainer = trainer.Trainer(config)
  
    if args.dataset_type == "replica":
        print("----- Replica Dataset -----")
        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))
        test_ids = [x+step//2 for x in train_ids]
        if args.training_mode:
            test_ids = [0]
        config["experiment"]["train_ids"] = train_ids
        config["experiment"]["test_ids"] = test_ids
        replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"], enable_fea=args.load_dino, fea_dim=config["experiment"]["feature_dim"], high_res_dino=args.high_res_dino, enable_lseg=args.load_lseg)
        print("--------------------")
        if args.label_propagation:
            if args.sparse_views: # add view-point sampling to partial sampling
                print("Sparse Viewing Labels Mode under ***Patial Labelling***! Sparse Ratio is ", args.sparse_ratio)
                replica_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)
            print("Label Propagation Mode! Partial labelling percentage is: {} ".format(args.partial_perc))
            replica_data_loader.simulate_user_click_partial(perc=args.partial_perc, load_saved=args.load_saved, visualise_save=args.visualise_save)

        elif args.user_click:
            if args.sparse_views: # add view-point sampling to partial sampling
                print("Sparse Viewing Labels Mode under ***Patial Labelling***! Sparse Ratio is ", args.sparse_ratio)
                replica_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)
            print("User Clicking Mode! Num Click is: {} ".format(args.num_click))
            replica_data_loader.simulate_user_clicks(num_click=args.num_click, load_saved=args.load_saved, visualise_save=args.visualise_save)

        elif args.sparse_views:
            if len(args.label_map_ids)>0:
                print("Use label maps only for selected frames, ", args.label_map_ids)
                replica_data_loader.sample_specific_labels(args.label_map_ids, train_ids)
            else:
                print("Sparse Labels Mode! Sparsity Ratio is ", args.sparse_ratio)
                replica_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)

        else:
            print("Standard setup with full dense supervision.")
        nerf_trainer.set_params_replica()
        nerf_trainer.prepare_data_replica(replica_data_loader)

    elif args.dataset_type == "scannet":
        print("----- ScanNet Dataset with NYUv2-40 Conventions-----")
        print("processing ScanNet scene: ", os.path.basename(config["experiment"]["dataset_dir"]))
        scannet_data_loader = scannet_datasets.ScanNet_Dataset( scene_dir=config["experiment"]["dataset_dir"],
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"],
                                                                    sample_step=config["experiment"]["sample_step"],
                                                                    save_dir=config["experiment"]["dataset_dir"],
                                                                    enable_fea=args.load_dino, fea_dim=config["experiment"]["feature_dim"], high_res_dino=args.high_res_dino)
        print("--------------------")
        if args.label_propagation:
            if args.sparse_views:
                print("Sparse Viewing Labels Mode! Sparse Ratio is ", args.sparse_ratio)
                scannet_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)
            print("Partial Segmentation Mode! Partial percentage is: {} ", args.partial_perc)
            scannet_data_loader.simulate_user_click_partial(perc=args.partial_perc, load_saved=args.load_saved, visualise_save=args.visualise_save, instance=args.test_instance)

        elif args.sparse_views:
            print("Sparse Viewing Labels Mode! Sparse Ratio is ", args.sparse_ratio)
            scannet_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)

        nerf_trainer.set_params_scannet(scannet_data_loader)
        nerf_trainer.prepare_data_scannet(scannet_data_loader)

    # Create nerf model, init optimizer
    start = nerf_trainer.create_nerf()
    # Create rays in world coordinates
    nerf_trainer.init_rays()

    # Short-cut for render only
    if args.render_only:
        print("RENDER ONLY")
        nerf_trainer.render_only('test', save_idx='step_200000')
        print('done')
        return

    if args.propagate_3d and args.ckpt_path is not None:
        sem_train, sem, mask, points_label, jacobian_train, agg = nerf_trainer.get_given_labels_jacobians()
        tree = jacobian_train
        if args.merge_instance:
            points_all_label = np.concatenate([label*np.ones(jacobian_train[i].shape[0], dtype=np.uint8) for i, label in enumerate(points_label)], 0)

        # render test-view images
        print("RENDER TEST")
        with torch.no_grad():
            if args.merge_instance and not args.train_agg:
                label_map = points_all_label
            else:
                label_map = points_label
            nerf_trainer.render_propagate(tree, points_label, 'test', label_map=label_map, agg=agg)
        return

    if args.perturb_gradient and args.ckpt_path is not None:
        print('perturb gradient!!')
        nerf_trainer.perturb_gradient_render(nerf_trainer.rays_vis, nerf_trainer.H_scaled, nerf_trainer.W_scaled, t=args.perturb_t, t_iter=args.t_iter, perturb_r=args.perturb_r, perturb_g=args.perturb_g, perturb_b=args.perturb_b)
        return

    if args.propagate_2d and args.ckpt_path is not None:
        sem_train, sem, mask, points_label, jacobian_train, agg_trainer = nerf_trainer.get_given_labels_jacobians()
        if args.merge_instance:
            points_label_all = []
        # propagate label with perturbing respond
        difference_maps = []
        rgb_difference_maps = []
        with torch.no_grad():
            if args.spatial:
                rgbs_g, disps_g, deps_g, vis_deps_g = nerf_trainer.render_path(nerf_trainer.rays_vis, save_dir=os.path.join(args.save_dir, "train_render"), idx=0, save_img=True)
            rgbs, disps, deps, vis_deps = nerf_trainer.render_path(nerf_trainer.rays_test, os.path.join(args.save_dir, "unperturbed_test"), save_img=True)

        for idx, grad in enumerate(tqdm(jacobian_train)):
            if args.mean_gradient:
                net_fine_copy = copy.deepcopy(nerf_trainer.nerf_net_fine)
                net_coarse_copy = copy.deepcopy(nerf_trainer.nerf_net_coarse)
                nerf_trainer.perturb_one_direction(grad, t=args.perturb_t)
                with torch.no_grad():
                    rgbs_p, disps_p, deps_p, vis_deps_p = nerf_trainer.render_path(nerf_trainer.rays_test, None, save_img=False)
                if args.no_abs:
                    rgbs_difference = np.mean((rgbs_p - rgbs), -1)
                else:
                    rgbs_difference = np.mean(abs(rgbs_p - rgbs), -1)
                nerf_trainer.nerf_net_fine = net_fine_copy
                nerf_trainer.nerf_net_coarse = net_coarse_copy
            elif args.mean_response or args.gradient_kmeans or args.adaptive_selection:
                # TODO: mean response (before normalization) or mean logits (after normalization)
                rgbs_difference = []
                for g in grad:
                    net_fine_copy = copy.deepcopy(nerf_trainer.nerf_net_fine)
                    net_coarse_copy = copy.deepcopy(nerf_trainer.nerf_net_coarse)
                    nerf_trainer.perturb_one_direction(g, t=args.perturb_t)
                    with torch.no_grad():
                        rgbs_p, disps_p, deps_p, vis_deps_p = nerf_trainer.render_path(nerf_trainer.rays_test, None, save_img=False)
                    if args.no_abs:
                        difference = np.mean((rgbs_p - rgbs), -1)
                    else:
                        difference = np.mean(abs(rgbs_p - rgbs), -1)
                    rgbs_difference.append(difference)
                    if args.merge_instance:
                        rgb_difference_maps.append(difference)
                        points_label_all.append(points_label[idx])
                    nerf_trainer.nerf_net_fine = net_fine_copy
                    nerf_trainer.nerf_net_coarse = net_coarse_copy
                rgbs_difference = np.stack(rgbs_difference, -1)
                rgbs_difference = np.mean(rgbs_difference, -1)
            if not args.merge_instance:
                rgb_difference_maps.append(rgbs_difference)

            if not args.train_agg:
                difference_map = []
                for idx_test, difference in enumerate(rgbs_difference):
                    # threshold
                    difference[difference > 0.8] = 0.8
                    difference = cv2.GaussianBlur(difference, (3, 3), 0)
                    difference = difference / (np.max(difference) + 1e-7)
                    # exclude the case when this label doesn't appear in this view
                    # print(np.std(difference), np.mean(difference))
                    if np.std(difference) < 0.05 and np.mean(difference) < 0.01:
                        difference = np.zeros_like(difference)
                    difference_map.append(difference)
                    # os.makedirs(os.path.join(args.save_dir, "perturbed_test_" + str(idx)), exist_ok=True)
                    # plt.clf()
                    # sns.heatmap(difference)
                    # plt.savefig(os.path.join(args.save_dir, "perturbed_test_" + str(idx), str(idx_test) + "_heatmap.png"))

                difference_map = np.stack(difference_map, 0)
                difference_maps.append(difference_map)

        if args.train_agg:
            rgb_difference_maps = np.stack(rgb_difference_maps, -1)
            print("rgb_difference_maps:", rgb_difference_maps.shape)
        else:
            difference_maps = np.stack(difference_maps, -1)
            difference_maps = softmax(difference_maps, axis=-1)

        if args.spatial:
            depth_g = deps_g[0]
            depth_g[sem_train == 0] = 0
            points_g = back_project(depth_g, nerf_trainer.K, np.linalg.inv(nerf_trainer.train_Ts[0].cpu().numpy()))
            pts = []
            for index, depth in enumerate(deps):
                sub_pts = back_project(depth, nerf_trainer.K, nerf_trainer.test_Ts[index].cpu().numpy())
                pts.append(sub_pts.reshape((nerf_trainer.H_scaled, nerf_trainer.W_scaled, 3)))
            pts = np.stack(pts, 0)
            distance_maps = []
            for idx_test, point_g in enumerate(points_g):
                distance_map = 1/(np.linalg.norm(pts - point_g, axis=-1)+1e-7)
                distance_maps.append(distance_map)
            distance_maps = np.stack(distance_maps, -1)
            distance_maps = softmax(distance_maps, axis=-1)
            difference_maps = np.multiply(difference_maps, distance_maps)

        if args.train_agg:
            responses = torch.tensor(rgb_difference_maps.reshape((-1, len(points_label)))).cuda()
            with torch.no_grad():
                segmentation_logits = agg_trainer.agg_net(responses).cpu().numpy()
            segmentation_logits = segmentation_logits.reshape((nerf_trainer.num_test, nerf_trainer.H_scaled, nerf_trainer.W_scaled, len(points_label)))
            labels_idx_agg = np.argmax(softmax(segmentation_logits, axis=-1), axis=-1)
            labels_idx_agg = labels_idx_agg[np.newaxis, ...]
            if args.merge_instance:
                labels_agg = points_label_all[labels_idx_agg].squeeze()
            else:
                labels_agg = points_label[labels_idx_agg].squeeze()
            labels_idx = np.argmax(softmax(rgb_difference_maps, axis=-1), axis=-1)
        else:
            labels_idx = np.argmax(softmax(difference_maps, axis=-1), axis=-1)
        labels_idx = labels_idx[np.newaxis, ...]

        if args.merge_instance:
            labels = points_label_all[labels_idx].squeeze()
        else:
            labels = points_label[labels_idx].squeeze()
        for i, label in enumerate(labels):
            if args.test_instance:
                gt_label = nerf_trainer.test_instance_scaled[i, ...].squeeze()
            else:
                gt_label = nerf_trainer.test_semantic_scaled[i, ...].squeeze()
            if args.train_agg:
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_label.png'), labels_agg[i].astype(np.uint8))
                error_map = np.where(np.abs(labels_agg[i] - gt_label) > 0, 255, 0)
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_error_map.png'), error_map)
                # vis_label = nerf_trainer.valid_colour_map.cpu().numpy()[labels_agg[i].astype(np.uint8)]
                # cv2.imwrite(os.path.join(args.save_dir, str(i) + '_vis_label.png'), vis_label)

                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_label_wo.png'), label.astype(np.uint8))
                error_map = np.where(np.abs(label - gt_label) > 0, 255, 0)
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_error_map_wo.png'), error_map)
                # vis_label = nerf_trainer.valid_colour_map.cpu().numpy()[label.astype(np.uint8)]
                # cv2.imwrite(os.path.join(args.save_dir, str(i) + '_vis_label_wo.png'), vis_label)
            else:
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_label.png'), label.astype(np.uint8))
                error_map = np.where(np.abs(label - gt_label) > 0, 255, 0)
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_error_map.png'), error_map)
                vis_label = nerf_trainer.valid_colour_map.cpu().numpy()[label.astype(np.uint8)]
                cv2.imwrite(os.path.join(args.save_dir, str(i) + '_vis_label.png'), vis_label)
        return

    N_iters = int(float(config["train"]["N_iters"])) + 1
    global_step = start
    ##########################
    if args.visualize_sample:
        if nerf_trainer.no_batching:
            training_vis = np.ones(nerf_trainer.num_train, nerf_trainer.H*nerf_trainer.W)
        else:
            training_vis = np.ones(nerf_trainer.num_train*nerf_trainer.H*nerf_trainer.W)
    print('Begin')
    #####  Training loop  #####
    for i in trange(start, N_iters):

        time0 = time.time()
        if args.visualize_sample:
            sampled_idx = nerf_trainer.step(global_step)
            if nerf_trainer.no_batching:
                index_batch, index_hw = sampled_idx
                training_vis[index_batch, index_hw] = 0
            else:
                training_vis[sampled_idx] = 0
        else:
            nerf_trainer.step(global_step)
        dt = time.time()-time0
        print()
        print("Time per step is :", dt)
        global_step += 1

    if args.visualize_sample:
        training_vis = training_vis.reshape(nerf_trainer.num_train, nerf_trainer.H, nerf_trainer.W)
        for img_idx, img in enumerate(training_vis):
            img = img*255
            cv2.imwrite(os.path.join(args.save_dir, 'vis_sample_'+str(img_idx)+'.png'), img)
    print('done')

if __name__=='__main__':
    train()