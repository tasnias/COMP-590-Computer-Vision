# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import numpy as np
import cv2


# -------------------------------------------------------------------------------


class FeatureMatcher:
    def __init__(self, args):
        self.window_size = args.matching_window_size

        if self.window_size % 2 != 1:
            raise ValueError("window size must be an odd number")

        if args.matching_method.lower() == "ssd":
            self.matching_method = self.match_ssd
        elif args.matching_method.lower() == "ncc":
            self.matching_method = self.match_ncc
        else:
            raise ValueError("invalid matching method")

    # ---------------------------------------------------------------------------

    # extract descriptors and match keypoints between two images
    #
    # inputs:
    # - image1: first input image, assumed to be grayscale
    # - keypoints1: N1 x 2 array of keypoint (x,y) pixel locations in the first
    #   image, assumed to be integer coordinates
    # - image2: second input image, assumed to be grayscale
    # - keypoints2: N2 x 2 array of keypoint (x,y) pixel locations in the second
    #   image, assumed to be integer coordinates
    # returns:
    # - matches: M x 2 array of indices for the matches; the first column
    #   provides the index for the keypoint in the first image, and the second
    #   column provides the corresponding keypoint index in the second image
    def __call__(self, image1, keypoints1, image2, keypoints2):
        d1 = self.get_descriptors(image1, keypoints1)
        d2 = self.get_descriptors(image2, keypoints2)

        match_matrix = self.matching_method(d1, d2)
        matches = self.compute_matches(match_matrix)

        return matches

    # ---------------------------------------------------------------------------

    # extract descriptors from an image
    #
    # inputs:
    # - image: input image, assumed to be grayscale
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the image,
    #   assumed to be integer coordinates
    # returns:
    # - descriptors: N x <window size**2> array of feature descriptors for the
    #   keypoints; in the implementation here, the descriptors are the
    #   (window_size, window_size) patch centered at every keypoint
    def get_descriptors(self, image, keypoints):
        n = int((self.window_size - 1) / 2)
        descriptor = np.empty((keypoints.shape[0], self.window_size ** 2))

        # Use replicate padding to take care of key points on the edge
        paddedim = cv2.copyMakeBorder(image, n, n, n, n, cv2.BORDER_REPLICATE)
        for i in range(keypoints.shape[0]):
            descriptor[i] = np.array(paddedim[keypoints[i, 1] - n + n:keypoints[i, 1] + n + n + 1,
                                     keypoints[i, 0] - n + n:keypoints[i, 0] + n + n + 1].flatten())
        return np.array(descriptor)

    # ---------------------------------------------------------------------------

    # compute a distance matrix between two sets of feature descriptors using
    # sum-of-squares differences
    #
    # inputs:
    # - d1: N1 x <feature_length> array of keypoint descriptors
    # - d2: N2 x <feature_length> array of keypoint descriptors
    # returns:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to d1 and the columns corresponding to d2
    def match_ssd(self, d1, d2):
        match_matrix = np.zeros((d1.shape[0], d2.shape[0]))
        for f1 in range(d1.shape[0]):
            for f2 in range(d2.shape[0]):
                match_matrix[f1, f2] = np.sum((d1[f1] - d2[f2]) ** 2)
        return match_matrix

    # ---------------------------------------------------------------------------

    # compute a distance matrix between two sets of feature descriptors using
    # one minus the normalized cross-correlation
    #
    # inputs:
    # - d1: N1 x <feature_length> array of keypoint descriptors
    # - d2: N2 x <feature_length> array of keypoint descriptors
    # returns:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to d1 and the columns corresponding to d2
    def match_ncc(self, d1, d2):
        match_matrix = np.zeros((d1.shape[0], d2.shape[0]))
        for f1 in range(d1.shape[0]):
            for f2 in range(d2.shape[0]):
                m1 = np.mean(d1[f1])
                m2 = np.mean(d2[f2])
                num = (d1[f1] - m1) * (d2[f2] - m2)
                dem = np.std(d1[f1]) * np.std(d2[f2])
                match_matrix[f1, f2] = 1 - np.sum(num / dem)
        return match_matrix

    # ---------------------------------------------------------------------------

    # given a matrix of descriptor distances for keypoint pairs, compute
    # keypoint correspondences between two images
    #
    # inputs:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to the N1 keypoints in the first image and the columns
    #   corresponding to the N2 keypoints in the second image
    # returns:
    # - matches: M x 2 array of indices for the M matches; the first column
    #   provides the index for the keypoint in the first image, and the second
    #   column provides the corresponding keypoint index in the second image
    def compute_matches(self, match_matrix):
        matches1 = []
        for idx1 in range(match_matrix.shape[0]):
            idx2 = np.argmin(match_matrix[idx1])
            matches1.append((idx1, idx2))

        matches2 = []
        for idx2 in range(match_matrix.shape[1]):
            idx1 = np.argmin(match_matrix[:, idx2])
            matches2.append((idx1, idx2))

        one_one = []
        for row in matches1:
            if row in matches2:
                one_one.append([row[0], row[1]])
        return np.array(one_one)
