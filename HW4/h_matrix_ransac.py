# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: RANSAC

import numpy as np
import random
from h_matrix import estimate_homography, compute_h_matrix_inliers
import cv2
import math

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)


# -------------------------------------------------------------------------------

# Run RANSAC on a dataset for a given model type
#
# @param data                 M x K numpy array containing all M observations in
#                             the dataset, each with K dimensions
# @param inlier_threshold     Given some error function for the model (e.g.,
#                             point-to-plane distance if the model is a 3D
#                             plane), label input data as inliers if their error
#                             is smaller than this threshold
# @param confidence_threshold Our chosen value p that determines the minimum
#                             confidence required to stop RANSAC
# @param max_num_trials       Initial maximum number of RANSAC iterations N
#
# @return best_model          The associated best model for the inliers
# @return inlier_mask         Length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
#                             otherwise; inliers can be recovered by taking
#                             data[inlier_mask]
def ransac(data, inlier_threshold, confidence_threshold, max_num_trials):
    max_iter = max_num_trials  # current maximum number of trials
    iter_count = 0  # current number of iterations

    S = 4

    best_inlier_count = 0  # initial number of inliers is zero
    best_inlier_mask = np.zeros(  # initially mark all samples as outliers
        len(data), dtype=np.bool)

    keypoints1 = np.column_stack((data[:, :2], np.ones(len(data))))
    keypoints2 = np.column_stack((data[:, 2:], np.ones(len(data))))

    # continue while the maximum number of iterations hasn't been reached
    while iter_count < max_iter:
        iter_count += 1

        # -----------------------------------------------------------------------
        # 1) sample as many points from the data as are needed to fit the
        #    relevant model
        correspondences = []
        correspondence_idxs = random.sample(range(0,len(data)), S)

        # Choose 4 random correspondences
        for idx in correspondence_idxs:
            correspondences.append(data[idx])
        correspondences = np.array(correspondences)

        # -----------------------------------------------------------------------
        # 2) fit a model to the sampled data subset
        # Calculate homography with these 4 points
        H = estimate_homography(correspondences)

        # -----------------------------------------------------------------------
        # 3) determine the inliers to the model; store the result as a boolean
        #    mask, with inliers referenced by data[inlier_mask]
        inlier_mask, inlier_count = compute_h_matrix_inliers(keypoints1, keypoints2, H, inlier_threshold)

        # -----------------------------------------------------------------------
        # 4) if this model is the best one yet, update the report and the
        #    maximum iteration threshold
        if inlier_count > best_inlier_count:
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            inlier_ratio = best_inlier_count / len(data)
            max_iter = math.log(1 - confidence_threshold) / math.log(1 - inlier_ratio**S)

    # ---------------------------------------------------------------------------
    # 5) run a final fit on the H matrix using the inliers
    # ---------------------------------------------------------------------------
    best_H = estimate_homography(data[best_inlier_mask])

    # print some information about the results of RANSAC
    print("Iterations:", iter_count)
    print("Inlier Ratio: {:.3f}".format(inlier_ratio))
    print("Best Fit Model:")
    print(best_H)

    return best_H, best_inlier_mask
