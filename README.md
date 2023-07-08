# JacobiNeRF: NeRF Shaping with Mutual Information Gradient

### [Paper](https://arxiv.org/abs/2304.00341) (CVPR 2023) | [Project Page](https://xxm19.github.io/jnerf/) | [Video](https://www.youtube.com/watch?v=uKU9UdVL6GQ)
[Xiaomeng Xu](https://xxm19.github.io/), [Yanchao Yang](https://yanchaoyang.github.io/), [Kaichun Mo](https://kaichun-mo.github.io/), [Boxiao Pan](https://cs.stanford.edu/~bxpan/), [Li Yi](https://ericyi.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)

### Dependencies
Main python dependencies are listed below:
- python>=3.8
- torch>=1.12.0
- cudatoolkit>=11.3

Requirements are listed in requirements.txt

## Datasets

#### Download dataset
We mainly use [Replica](https://github.com/facebookresearch/Replica-Dataset) and [ScanNet](http://www.scan-net.org/) datasets for experiments, where we train a plain NeRF model and Jacobi-NeRF model on each 3D scene.

We also use [pre-rendered Replica data](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) provided by Semantic-NeRF.

#### Extract DINO feature
Extract per-pixel DINO feature from training images first. We use dino_vits8, layer 11 to create descriptors from. Please download the code of the [DINO repo](https://github.com/ShirAmir/dino-vit-features) and run: 
```
python extractor.py --model dino_vits8 --load_size 360 --image_path [image_path] --output_path [dino_path]
```

You can add a simple for loop in the [DINO repo](https://github.com/ShirAmir/dino-vit-features) to extract DINO features for all images at once.

#### Dataset directory
Set ```dataset_dir``` in config_file```./configs/replica_room0_config.yaml)```, structured as follows.
```
  dataset_dir
  ├── rgb 	# RGB images
  ├── semantic_class    # ground truth semantic segmentation labels
  ├── dino  # DINO features
  ├── traj_w_c.txt  # camera parameters
  └── ...
 ```

## Running code

#### plain NeRF training
- Set ```N_iters``` in config_file```./configs/replica_room0_config.yaml``` as 200000.
```
python main.py --config_file ./configs/replica_room0_config.yaml --save_dir [save_dir] --training_mode
```

#### Jacobi-NeRF training
- Set ```N_iters``` in config_file```./configs/replica_room0_config.yaml``` as 10000.
```
python main.py --config_file ./configs/replica_room0_config.yaml --load_dino --load_on_cpu --feature_dim 384 --save_dir [save dir] --ckpt_path [plain nerf ckpt path] --contrastive_2d --contrastive_starting_step 0 --contrastive_step 1 --wgt_img 1 --wgt_contrastive 1e-2 --wgt_gradient 1e-2 --rgb_layer --pos_threshold 0.8 --neg_threshold 0.8 --adaptive_threshold --gt_feature_sim --gradient_norm_loss --N_rays 64 --contrastive_abs --training_mode
```

#### 3D label propagation
- sparse setting
```
python main.py --config_file ./configs/replica_room0_config.yaml --sparse_views --sparse_ratio 0.995 --label_propagation --partial_perc 0 --load_saved --propagate_3d --ckpt_path [J-NeRF ckpt path] --save_dir [save_dir] --rgb_layer --mean_gradient
```
- dense setting
```
python main.py --config_file ./configs/replica_room0_config.yaml --sparse_views --sparse_ratio 0.995 --propagate_3d --ckpt_path [J-NeRF ckpt path] --save_dir [save_dir] --rgb_layer --adaptive_selection --num_comb 20 --num_iters 5 --train_agg
```

#### 2D label propagation
- sparse setting
```
python main.py --config_file ./configs/replica_room0_config.yaml --sparse_views --sparse_ratio 0.995 --label_propagation --partial_perc 0 --load_saved --propagate_2d --ckpt_path [J-NeRF ckpt path] --save_dir [save_dir] --rgb_layer --mean_gradient
```
- dense setting
```
python main.py --config_file ./configs/replica_room0_config.yaml --sparse_views --sparse_ratio 0.995 --propagate_2d --ckpt_path [J-NeRF ckpt path] --save_dir [save_dir] --rgb_layer --adaptive_selection --num_comb 20 --num_iters 5 --train_agg
```

## Acknowledgements
Thanks [nerf](https://github.com/bmild/nerf), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf) for nice implementation of NeRF and scripts in processing datasets.

## Citation
If you find JacobiNeRF useful for your work, please cite:
```
@inproceedings{xu2023jacobinerf,
  title={JacobiNeRF: NeRF Shaping with Mutual Information Gradients},
  author={Xu, Xiaomeng and Yang, Yanchao and Mo, Kaichun and Pan, Boxiao and Yi, Li and Guibas, Leonidas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16498--16507},
  year={2023}
}
```

## Contact
If you have any questions, please contact xiaomengxu0830@gmail.com.

