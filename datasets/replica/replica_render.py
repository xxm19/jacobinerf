import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import habitat_sim.registry as registry
# from transforms3d import quaternions
import quaternion

from habitat_sim.utils.data import ImageExtractor, PoseExtractor

@registry.register_pose_extractor(name="panorama_pose_extractor")
class PanoramaExtractor(PoseExtractor):
    def extract_poses(self, view, fp: str):
        # Determine the physical spacing between each camera position
        height, width = view.shape
        dist = min(height, width) // 30  # We can modify this to be user-defined later

        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        # Exclude camera positions at invalid positions
        gridpoints = []
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, view):
                    gridpoints.append(point)

        # Find the closest point of the target class to each gridpoint
        poses = []
        for point in gridpoints:
            point_label_pairs = self._panorama_extraction(point, view, dist)
            poses.extend([(point, point_, fp) for point_, label in point_label_pairs])

        # Returns poses in the coordinate system of the topdown view
        return poses

    def _panorama_extraction(
        self, point, view, dist):
        in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
            view
        ) and 0 <= col < len(view[0])
        point_label_pairs = []
        r, c = point
        neighbor_dist = dist // 2
        neighbors = [
            (r - neighbor_dist, c - neighbor_dist),
            (r - neighbor_dist, c),
            (r - neighbor_dist, c + neighbor_dist),
            (r, c - neighbor_dist),
            (r, c + neighbor_dist),
            (r + neighbor_dist, c - neighbor_dist),
            # (r + step, c), # Exclude the pose that is in the opposite direction of habitat_sim.geo.FRONT, causes the quaternion computation to mess up
            (r + neighbor_dist, c + neighbor_dist),
        ]

        for n in neighbors:
            # Only add the neighbor point if it is navigable. This prevents camera poses that
            # are just really close-up photos of some object
            if in_bounds_of_topdown_view(*n) and self._valid_point(*n, view):
                point_label_pairs.append((n, 0.0))

        return point_label_pairs

@registry.register_pose_extractor(name="random_pose_extractor")
class RandomPoseExtractor(PoseExtractor):
    def extract_poses(self, view, fp):
        height, width = view.shape
        num_random_points = 900
        points = []
        while len(points) < num_random_points:
            # Get the row and column of a random point on the topdown view
            row, col = np.random.randint(int(height/4), int(height/2)), np.random.randint(int(width/4), int(width/2))
            # row, col = np.random.randint(0, height), np.random.randint(0, width)
            # Convenient method in the PoseExtractor class to check if a point
            # is navigable
            if self._valid_point(row, col, view):
                points.append((row, col))
        # print(points)
        points = sorted(points, reverse=True)
        # print(points)
        poses = []
        # Now we need to define a "point of interest" which is the point the camera will
        # look at. These two points together define a camera position and angle
        for point in points:
            r, c = point
            point_of_interest = (r - 1, c - 1)
            pose = (point, point_of_interest, fp)
            poses.append(pose)
        return poses

def save_sample(sample, idx, save_dir, instance_id_to_semantic_label_id):
    img = np.array(sample["rgba"]).astype(np.uint8)
    depth = np.array(sample["depth"]).astype(np.uint8)
    semantic_instance = np.array(sample["semantic"]).astype(np.uint8)
    img_path = os.path.join(save_dir, "rgb", "rgb_"+str(idx)+".png")
    depth_path = os.path.join(save_dir, "depth", "depth_"+str(idx)+".png")
    semantic_class_path = os.path.join(save_dir, "semantic_class", "semantic_class_"+str(idx)+".png")
    semantic_instance_path = os.path.join(save_dir, "semantic_instance", "semantic_instance_"+str(idx)+".png")
    os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "semantic_class"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "semantic_instance"), exist_ok=True)
    cv2.imwrite(img_path, img)
    cv2.imwrite(depth_path, depth)
    cv2.imwrite(semantic_instance_path, semantic_instance)
    semantic_class = instance_id_to_semantic_label_id[semantic_instance]
    cv2.imwrite(semantic_class_path, semantic_class)

scene_name = "frl_apartment_5"
sequence_name = "Sequence_1"
scene_filepath = os.path.join("/nas/xxm/replica_v1/", scene_name, "habitat", "mesh_semantic.ply")
save_dir = os.path.join("/nas/xxm/Replica_Dataset/", scene_name, sequence_name)
json_class_mapping = os.path.join("/nas/xxm/replica_v1/", scene_name, "habitat", "info_semantic.json")
with open(json_class_mapping, "r") as f:
    annotations = json.load(f)
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])

extractor = ImageExtractor(
    scene_filepath,
    pose_extractor_name="random_pose_extractor",
    img_size=(480, 640),
    output=["rgba", "depth", "semantic"],
    shuffle=False,
)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('full')

poses = extractor.poses
world_mats = []
for pose in poses:
    world_mat = np.eye(4)
    world_mat[:3, 3] = np.array(pose[0])
    world_mat[:3, :3] = quaternion.as_rotation_matrix(pose[1])
    world_mat[[1, 2]] = - world_mat[[1, 2]]  # opengl to opencv coordinate
    world_mats.append(world_mat)
os.makedirs(save_dir, exist_ok=True)
poses_path = os.path.join(save_dir, "traj_w_c.txt")
f = open(poses_path, "w")
for mat in world_mats:
    for i, e in enumerate(mat.reshape(-1)):
        f.write(str(e))
        if i < len(mat.reshape(-1)) - 1:
            f.write(' ')
    f.write('\n')
f.close()

for idx, sample in enumerate(extractor):
    save_sample(sample, idx, save_dir, instance_id_to_semantic_label_id)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()