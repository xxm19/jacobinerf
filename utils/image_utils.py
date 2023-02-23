import os
import cv2
import numpy as np
import imgviz
from imgviz import label_colormap
from imgviz import draw as draw_module
import matplotlib.pyplot as plt
import sklearn.decomposition as decompose
to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def numpy2cv(image):
    """

    :param image: a floating numpy images of shape [H,W,3] within range [0, 1]
    :return:
    """

    image_cv = np.copy(image)
    image_cv = np.astype(np.clip(image_cv, 0, 1)*255, np.uint8)[:, :, ::-1]  # uint8 BGR opencv format
    return image_cv

def vis_descriptor(descriptors, H=240, W=320):
    features = descriptors.cpu().numpy()
    print(features.shape)  # [H_out*W_out, feature_dim]
    if features.shape[-1] > 3:
        pca = decompose.PCA(3)
        features = pca.fit_transform(features)
    feature_viz = features.reshape(H, W, 3)
    print(feature_viz.shape)
    # dim = (320, 240)
    # resize_feature_viz = cv2.resize(feature_viz, dim, interpolation=cv2.INTER_AREA)
    return to8b_np(feature_viz)


def plot_semantic_legend(
    label, 
    label_name, 
    colormap=None, 
    font_size=30,
    font_path=None,
    save_path=None,
    img_name=None):


    """Plot Colour Legend for Semantic Classes

    Parameters
    ----------
    label: numpy.ndarray, (N,), int
        One-dimensional array containing the unique labels of exsiting semantic classes
    label_names: list of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
    Legend image of visualising semantic labels.

    """

    label = np.unique(label)
    if colormap is None:
        colormap = label_colormap()

    text_sizes = np.array(
            [
                draw_module.text_size(
                    label_name[l], font_size, font_path=font_path
                )
                for l in label
            ]
        )

    text_height, text_width = text_sizes.max(axis=0)
    legend_height = text_height * len(label) + 5
    legend_width = text_width + 20 + (text_height - 10)


    legend = np.zeros((legend_height+50, legend_width+50, 3), dtype=np.uint8)
    aabb1 = np.array([25, 25], dtype=float)
    aabb2 = aabb1 + (legend_height, legend_width)

    legend = draw_module.rectangle(
        legend, aabb1, aabb2, fill=(255, 255, 255)
    )  # fill the legend area by white colour

    y1, x1 = aabb1.round().astype(int)
    y2, x2 = aabb2.round().astype(int)

    for i, l in enumerate(label):
        box_aabb1 = aabb1 + (i * text_height + 5, 5)
        box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
        legend = draw_module.rectangle(
            legend, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
        )
        legend = draw_module.text(
            legend,
            yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
            text=label_name[l],
            size=font_size,
            font_path=font_path,
            )

    
    plt.figure(1)
    plt.title("Semantic Legend!")
    plt.imshow(legend)
    plt.axis("off")

    img_arr = imgviz.io.pyplot_to_numpy()
    plt.close()
    if save_path is not None:
        import cv2
        if img_name is not None:
            sav_dir = os.path.join(save_path, img_name)
        else:
            sav_dir = os.path.join(save_path, "semantic_class_Legend.png")
        # plt.savefig(sav_dir, bbox_inches='tight', pad_inches=0)
        cv2.imwrite(sav_dir, img_arr[:,:,::-1])
    return img_arr




def image_vis(
    pred_data_dict,
    gt_data_dict,
    # enable_sem = True
    ):
    to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    batch_size = pred_data_dict["vis_deps"].shape[0]

    gt_dep_row = np.concatenate(np.split(gt_data_dict["vis_deps"], batch_size, 0), axis=-2)[0]
    gt_raw_dep_row = np.concatenate(np.split(gt_data_dict["deps"], batch_size, 0), axis=-1)[0]

    gt_sem_row = np.concatenate(np.split(gt_data_dict["vis_sems"], batch_size, 0), axis=-2)[0]
    gt_sem_clean_row = np.concatenate(np.split(gt_data_dict["vis_sems_clean"], batch_size, 0), axis=-2)[0]
    gt_rgb_row = np.concatenate(np.split(gt_data_dict["rgbs"], batch_size, 0), axis=-2)[0]
        
    pred_dep_row = np.concatenate(np.split(pred_data_dict["vis_deps"], batch_size, 0), axis=-2)[0]
    pred_raw_dep_row = np.concatenate(np.split(pred_data_dict["deps"], batch_size, 0), axis=-1)[0]

    pred_sem_row = np.concatenate(np.split(pred_data_dict["vis_sems"], batch_size, 0), axis=-2)[0]
    pred_entropy_row = np.concatenate(np.split(pred_data_dict["vis_sem_uncers"], batch_size, 0), axis=-2)[0]
    pred_rgb_row = np.concatenate(np.split(pred_data_dict["rgbs"], batch_size, 0), axis=-2)[0]

    rgb_diff = np.abs(gt_rgb_row - pred_rgb_row)

    dep_diff = np.abs(gt_raw_dep_row - pred_raw_dep_row)
    dep_diff[gt_raw_dep_row== 0] = 0
    dep_diff_vis = imgviz.depth2rgb(dep_diff)

    views = [to8b_np(gt_rgb_row), to8b_np(pred_rgb_row), to8b_np(rgb_diff),
            gt_dep_row, pred_dep_row, dep_diff_vis,
            gt_sem_clean_row, gt_sem_row, pred_sem_row, pred_entropy_row]

    viz = np.vstack(views)
    return viz




nyu13_colour_code = (np.array([[0, 0, 0],
                       [0, 0, 1], # BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])*255).astype(np.uint8)


# color palette for nyu34 labels
nyu34_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
    #    (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
    #    (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
    #    (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
    #    (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
    #    (178, 127, 135),       # white board

    #    (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)



# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)

# instance color
instance_color_code = np.array([
    (136/255.0,224/255.0,239/255.0), (180/255.0,254/255.0,152/255.0), (184/255.0,59/255.0,94/255.0), (106/255.0,44/255.0,112/255.0),
    (39/255.0,53/255.0,135/255.0), (0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0),
    (162/255.0,210/255.0,255/255.0), (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (233/255.0, 69/255.0, 96/255.0),
    (234/255.0,255/255.0,208/255.0),(249/255.0,237/255.0,105/255.0), (255/255.0, 0, 99/255.0), (0, 255/255.0, 209/255.0),
    (66/255.0, 95/255.0, 87/255.0), (225/255.0, 77/255.0, 42/255.0), (0, 18/255.0, 83/255.0),
    (255/255.0, 115/255.0, 29/255.0), (95/255.0, 157/255.0, 247/255.0), (255/255.0, 248/255.0, 10/255.0), (214/255.0, 28/255.0, 78/255.0), (58/255.0, 176/255.0, 255/255.0),
    (85/255.0, 73/255.0, 148/255.0), (246/255.0, 117/255.0, 168/255.0), (29/255.0, 28/255.0, 229/255.0), (47/255.0, 143/255.0, 157/255.0), (198/255.0, 137/255.0, 198/255.0),
    (55/255.0, 226/255.0, 213/255.0), (251/255.0, 203/255.0, 10/255.0), (21/255.0, 0, 80/255.0), (66/255.0, 95/255.0, 87/255.0), (212/255.0, 217/255.0, 37/255.0), (180/255.0, 205/255.0, 230/255.0),
    (107/255.0, 114/255.0, 142/255.0), (220/255.0, 95/255.0, 0), (238/255.0, 238/255.0, 238/255.0), (135/255.0, 88/255.0, 255/255.0), (63/255.0, 167/255.0, 150/255.0), (161/255.0, 0, 53/255.0),
    (254/255.0, 194/255.0, 96/255.0), (42/255.0, 9/255.0, 68/255.0), (240/255.0, 255/255.0, 66/255.0), (255/255.0, 135/255.0, 135/255.0), (144/255.0, 161/255.0, 125/255.0), (255.0, 225.0, 225.0),
    (207/255.0, 10/255.0, 10/255.0), (29/255.0, 28/255.0, 229/255.0), (225/255.0, 77/255.0, 42/255.0), (0, 245/255.0, 255/255.0), (88/255.0, 169/255.0, 234/255.0), (0, 38/255.0, 161/255.0),
    (40/255.0, 14/255.0, 4/255.0), (66/255.0, 16/255.0, 89/255.0), (233/255.0, 66/255.0, 215/255.0)
], dtype=np.float32)
instance_color_code = instance_color_code*255
instance_color_code = instance_color_code.astype(np.uint8)

