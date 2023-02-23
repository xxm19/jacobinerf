import numpy as np
from scipy.spatial import KDTree
import cv2

def back_project(depth, intrinsic, extrinsic):
    v, u = np.where(depth > 0)
    uv = np.vstack((u + 0.5, v + 0.5, np.ones(u.shape[0])))
    uv = np.matmul(np.linalg.inv(intrinsic), uv)
    cp = uv * np.reshape(depth[depth > 0], (1, -1))
    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3:4]
    r_inv = np.linalg.inv(r)
    wp = np.matmul(r_inv, cp - t).transpose()
    return wp

def project(pts, intrinsic, extrinsic):
    # return cv2.projectPoints(pts, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3:4], intrinsic, None)[0]
    pts = pts.reshape(3, -1)
    projection_mat = intrinsic @ extrinsic[:3, ...]
    pixels = np.dot(projection_mat, np.append(pts, np.ones((1, pts.shape[1])), axis=0))
    pixels[0, :] = pixels[0, :] / pixels[2, :]
    pixels[1, :] = pixels[1, :] / pixels[2, :]
    pixels = np.delete(pixels, 2, 0)
    return pixels

def visualize_pairs(pts_pairs, intrinsic, extrinsic, img, color=(0, 255, 0), save_path=None):
    # pts_pairs [points_num, 3, 2]
    for i in range(pts_pairs.shape[0]):
        pts_pair = project(pts_pairs[i, ...], intrinsic, extrinsic)     # [2, 2]
        cv2.line(img, (int(pts_pair[0, 0]), int(pts_pair[0, 1])), (int(pts_pair[1, 0]), int(pts_pair[1, 1])), color, 1)
    cv2.imwrite(save_path, img)

def from2dto3d(List2D, K, R, t, d=1.75):
    # List2D : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.
    # d : depth?

    List2D = np.array(List2D)
    List3D = []
    # t.shape = (3,1)

    for p in List2D:
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T
        p.shape = (3, 1)
        # print("pixel: \n", p)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p
        # print("pc : \n", pc, pc.shape)

        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)
        # print("pw : \n", pw, t.shape, R.shape, pc.shape)

        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0]).T
        cam.shape = (3, 1)
        cam_world = t + R @ cam
        # print("cam_world : \n", cam_world)

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print("unit_vector : \n", unit_vector)

        # Point scaled along this ray
        p3D = cam_world + d * unit_vector
        # print("p3D : \n", p3D)
        List3D.append(p3D)

    return List3D

def build_KDTree(points):
    # points: array_like, (n, m), the n data points of dimension m to be indexed
    tree = KDTree(points)
    return tree

def calc_nearest_KDTree(tree, query):
    d, i = tree.query(query, 1)
    # print(d, i)
    return d, i