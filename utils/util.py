import torch
import numpy as np
import torch.nn as nn

def compute_entropy(x):
    p_x = torch.softmax(x, dim=0)
    entropy = p_x * torch.log(p_x)
    return -torch.sum(entropy, dim=0)


def show_point_clouds(pts, out):
    fout = open(out, 'w')
    MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
    pts[:, 3:6] = pts[:, 3:6] * 256.0 + MEAN_COLOR_RGB
    for i in range(pts.shape[0]):
        fout.write('v %f %f %f %d %d %d\n' % (
            pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3], pts[i, 4], pts[i, 5]))
    fout.close()


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d
