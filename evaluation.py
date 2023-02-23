import numpy as np
import cv2
import yaml
import os
import argparse
from training.training_utils import calculate_segmentation_metrics, calculate_depth_metrics
from datasets.replica import replica_datasets
from datasets.scannet import scannet_datasets

img2mse = lambda x, y: np.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * np.log(x) / np.log(np.array([10.]))

def evaluate(label_dir, ids, save_vis_label=False, save_error_map=False, test_given_labels=True, sparse_views=True, sparse_ratio=0.995, label_propagation=True, partial_perc=0, dataset_type='replica', config=None, instance=False, psnr=False):
    if dataset_type == 'replica':
        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))
        test_ids = [x + step // 2 for x in train_ids]
        replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                   train_ids=train_ids, test_ids=test_ids,
                                                                   img_h=config["experiment"]["height"],
                                                                   img_w=config["experiment"]["width"])
        if sparse_views:
            replica_data_loader.sample_label_maps(sparse_ratio=sparse_ratio, random_sample=False, load_saved=True)
        # if label_propagation:
        #     replica_data_loader.simulate_user_click_partial(perc=partial_perc, load_saved=True, visualise_save=False)
        num_semantic_class = replica_data_loader.num_semantic_class  # number of semantic classes, including void class=0
        num_class = num_semantic_class - 1  # exclude void class
        mask_ids = replica_data_loader.mask_ids.astype(np.bool)
        # num_class = num_semantic_class
        test_samples = replica_data_loader.test_samples
        if test_given_labels:
            train_samples = replica_data_loader.train_samples
            train_semantic = train_samples["semantic_remap"][mask_ids].astype(np.int8) - 1
            train_semantic = train_semantic[train_semantic >= 0]
        test_image = test_samples["image"]  # [num_test, H, W, 3]
        color_map = replica_data_loader.colour_map_np[1:, :]
        # color_map = replica_data_loader.colour_map_np
        test_semantic = test_samples["semantic_remap"].astype(np.int8) - 1  # [num_test, H, W]
        # test_semantic = test_samples["semantic_remap"]  # [num_test, H, W]

    elif dataset_type == 'scannet':
        scannet_data_loader = scannet_datasets.ScanNet_Dataset(scene_dir=config["experiment"]["dataset_dir"],
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"],
                                                                    sample_step=config["experiment"]["sample_step"],
                                                                    save_dir=config["experiment"]["dataset_dir"])
        if sparse_views:
            scannet_data_loader.sample_label_maps(sparse_ratio=sparse_ratio, random_sample=False, load_saved=True)
        test_image = scannet_data_loader.test_samples["image"]  # [num_test, H, W, 3]
        mask_ids = scannet_data_loader.mask_ids.astype(np.bool)
        if instance:
            color_map = scannet_data_loader.instance_colour_map_np
            test_semantic = scannet_data_loader.test_samples["instance"]
            if test_given_labels:
                train_semantic = scannet_data_loader.train_samples["instance"][mask_ids]
                train_semantic = train_semantic[train_semantic >= 0]
            num_class = scannet_data_loader.num_instance
        else:
            num_semantic_class = scannet_data_loader.num_semantic_class  # number of semantic classes, including void class=0
            num_class = num_semantic_class - 1  # exclude void class
            test_samples = scannet_data_loader.test_samples
            if test_given_labels:
                train_samples = scannet_data_loader.train_samples
                train_semantic = train_samples["semantic_remap"][mask_ids].astype(np.int8) - 1
                train_semantic = train_semantic[train_semantic >= 0]
            color_map = scannet_data_loader.colour_map_np[1:, :]
            test_semantic = test_samples["semantic_remap"].astype(np.int8) - 1  # [num_test, H, W]
    else:
        raise NotImplementedError

    if psnr:
        psnrs = []
        mses = []
    else:
        sems = []
        # ignore_label = -1
        ignore_label = []
        ignore_label.append(-1)
        existing_class_mask = np.ones(num_class, dtype=np.bool)
        if test_given_labels:
            given_label = np.unique(train_semantic)
            print(given_label)
            for label in range(num_class):
                if label not in given_label:
                    existing_class_mask[label] = False
                    ignore_label.append(label)
        print("ignore_label:", ignore_label)
        print("existing_class_mask:", existing_class_mask)

    for idx in ids:
        if psnr:
            image = cv2.imread(os.path.join(label_dir, 'rgb_'+str(idx).rjust(3, '0')+'.png'), cv2.IMREAD_UNCHANGED)/255.0
            mse = img2mse(image, test_image[idx, ...])
            mses.append(mse)
            psnrs.append(mse2psnr(mse))
        else:
            # label = cv2.imread(os.path.join(label_dir, 'label_'+str(idx).rjust(3, '0')+'.png'), cv2.IMREAD_UNCHANGED)
            label = cv2.imread(os.path.join(label_dir, str(idx)+'_label.png'), cv2.IMREAD_UNCHANGED)
            if save_vis_label:
                vis_label = color_map[label]
                vis_gt_label = color_map[test_semantic[idx, ...]]
                cv2.imwrite(os.path.join(label_dir, str(idx)+'_vis_label.png'), vis_label)
                cv2.imwrite(os.path.join(label_dir, str(idx)+'_vis_gt_label.png'), vis_gt_label)
            if save_error_map:
                error_map = np.where(np.abs(label - test_semantic[idx, ...]) > 0, 255, 0)
                cv2.imwrite(os.path.join(label_dir, str(idx)+'_error_map.png'), error_map)
            sems.append(label)
    if psnr:
        psnr_loss = np.mean(psnrs)
        mse_loss = np.mean(mses)
        print("mse:", mse_loss)
        print("psnr:", psnr_loss)
    else:
        sems = np.stack(sems, 0)
        # test_semantic = test_semantic[60:100]
        miou, miou_validclass, total_accuracy, class_average_accuracy, ious, class_average_accuracy_validclass = \
            calculate_segmentation_metrics(true_labels=test_semantic, predicted_labels=sems,
                                           number_classes=num_class, ignore_label=ignore_label, class_mask=existing_class_mask)

        print('miou', miou)
        print('miou_validclass', miou_validclass)
        print('total_accuracy', total_accuracy)
        print('class_average_accuracy', class_average_accuracy)
        print('class_average_accuracy_validclass', class_average_accuracy_validclass)
        print('ious', ious)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default="/home/xiaomeng/semantic_nerf_configs/SSR_room1_config.yaml", help='config file name.')
    # parser.add_argument('--config_file', type=str, default="./SSR/configs/SSR_room1_config.yaml", help='config file name.')
    parser.add_argument('--config_file', type=str, default="/home/xiaomeng/semantic_nerf_configs/SSR_ScanNet_scene0004_00_config.yaml", help='config file name.')
    # parser.add_argument('--config_file', type=str, default="./SSR/configs/SSR_ScanNet_scene0480_01_config.yaml", help='config file name.')
    parser.add_argument('--ignore_labels', action='store_true')
    parser.add_argument('--instance', action='store_true')
    parser.add_argument('--psnr', action='store_true')
    parser.add_argument('--dataset_type', type=str, default="replica", choices= ["replica", "scannet"])
    parser.add_argument("--sparse_views", action='store_true',
                        help='Use labels from a sparse set of frames')
    parser.add_argument("--sparse_ratio", type=float, default=0.995,
                        help='The portion of dropped labelling frames during training, which can be used along with all working modes.')
    parser.add_argument("--label_propagation", action='store_true',
                        help='Label propagation using partial seed regions.')
    parser.add_argument("--partial_perc", type=float, default=0,
                        help='0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation')
    parser.set_defaults(ignore_labels=True, sparse_views=True, sparse_ratio=0.995)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    # label_dir = '/nas/xuxm/semantic_nerf_result/propagate_3d_0825_perc0_plain_renderdepth_room0_050000_spatial/test_render/'
    label_dir = '/share/xxm/propagation_result/Scan0004_00_propagate_3d_mi_1107_con_recon_gradnorm_batch64_adaptive_10000_0.995_dense/test_render/'
    ids = list(np.arange(0, 186))
    evaluate(label_dir, ids, save_vis_label=False, save_error_map=False, test_given_labels=args.ignore_labels, sparse_views=args.sparse_views, sparse_ratio=args.sparse_ratio, label_propagation=args.label_propagation, partial_perc=args.partial_perc, dataset_type=args.dataset_type, config=config, instance=args.instance, psnr=args.psnr)