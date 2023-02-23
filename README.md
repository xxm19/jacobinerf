# Jacobi-NeRF: NeRF Shaping with Mutual Information Gradient
Xiaomeng Xu, Yanchao Yang, Kaichun Mo, Boxiao Pan, Li Yi, Leonidas Guibas

### Dependencies
Main python dependencies are listed below:
- python>=3.8
- torch>=1.12.0
- cudatoolkit>=11.3

Requirements are listed in requirements.txt

## Datasets
We mainly use [Replica](https://github.com/facebookresearch/Replica-Dataset) and [ScanNet](http://www.scan-net.org/) datasets for experiments, where we train a plain NeRF model and Jacobi-NeRF model on each 3D scene.

We also use [pre-rendered Replica data](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) provided by Semantic-NeRF.

## Running code

#### plain NeRF training
```
python main.py --config_file ./configs/replica_room0_config.yaml --save_dir [save_dir] --training_mode
```

#### Jacobi-NeRF training
- Extract per-pixel [DINO](https://github.com/ShirAmir/dino-vit-features) feature from training images first. We use dino_vits8, layer 11 to create descriptors from.
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

## Contact
If you have any questions, please contact xiaomengxu0830@gmail.com.

