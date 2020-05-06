# Tasnia Sarwar
# COMP 590, Spring 590
# Assignment: RANSAC

import numpy as np
import math


# -------------------------------------------------------------------------------

# estimate a fundamental matrix from a set of correspondences
# here, we'll apply Hartley normalization, do a DLT, and enforce the rank-2
# constraint on the F matrix
# NOTE: Don't forget to pre-/post-multiply your final F matrix by the
#   Hartley transformations!
#
# @param correspondences      M x 4 numpy array of (x1, y1, x2, y2) matches
#
# @return F                   3x3 fundamental matrix
def estimate_homography(correspondences):
    T1, p1 = precondition(correspondences[:, :2])
    T2, p2 = precondition(correspondences[:, 2:])

    A = []
    for i in range(len(p1)):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        a1 = [x1, y1, 1, 0, 0, 0, - x1 * x2, -y1 * x2, -x2]
        a2 = [0, 0, 0, x1, y1, 1, - x1 * y2, -y2 * y1, -y2]
        A.append(a1)
        A.append(a2)
    A = np.asarray(A, dtype='float')
    u, d, uT = np.linalg.svd(A.T @ A)
    Hn = np.reshape(u[:, np.argmin(d)], (3, 3))
    H = np.linalg.inv(T2) @ Hn @ T1
    return H


# -------------------------------------------------------------------------------

# Given a set of 2D keypoint correspondences in a pair of images, and a
# hypothesized homography matrix relating the two images, compute which
# correspondences are inliers:
# 
#
# @param keypoints1           M x 3 array of (x,y,1) coordinates (first image)
# @param keypoints2           M x 3 array of (x,y,1) coordinates (second image)
# @param H                   3x3 homography matrix hypothesis
# @param inlier_threshold     given some error function for the model (e.g.,
#                             point-to-point distance )
# @return inlier_mask         length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
def compute_h_matrix_inliers(keypoints1, keypoints2, H, inlier_threshold):
    inlier_mask = np.zeros(len(keypoints1), dtype=np.bool)
    inlier_count = 0
    for p in range(len(keypoints1)):
        projected_point = H @ keypoints1[p]
        if projected_point[2] != 0 and math.sqrt(
                np.sum((keypoints2[p] - projected_point / projected_point[2]) ** 2)) <= inlier_threshold:
            inlier_count += 1
            inlier_mask[p] = True
    return inlier_mask, inlier_count


def precondition(points):
    p = np.zeros(tuple(points.shape), dtype='float32')
    # center of mass factor
    m = [np.sum(points[:, 0]) / len(points), np.sum(points[:, 1]) / len(points)]
    # scale factor
    s = [np.std(points[:, 0]) / math.sqrt(2), np.std(points[:, 1]) / math.sqrt(2)]
    # Transformation matrix
    T = [[1 / s[0], 0, -m[0] / s[0]], [0, 1 / s[1], -m[1] / s[1]], [0, 0, 1]]
    # Transformed point set
    p[:, 0] = (points[:,0] - m[0]) / s[0]
    p[:, 1] = (points[:,1] - m[1]) / s[1]
    return T, p
