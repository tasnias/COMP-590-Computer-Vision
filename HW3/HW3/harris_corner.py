# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter, sobel
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------


class HarrisCornerFeatureDetector:
    def __init__(self, args):
        self.gaussian_sigma = args.gaussian_sigma
        self.maxfilter_window_size = args.maxfilter_window_size
        self.harris_corner_k = args.harris_corner_k
        self.max_num_features = args.max_num_features

    # ---------------------------------------------------------------------------

    # detect corner features in an input image
    # inputs:
    # - image: a grayscale image
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the image,
    #   assumed to be integer coordinates
    def __call__(self, image):
        corner_response = self.compute_corner_response(image)
        keypoints = self.get_keypoints(corner_response)

        return keypoints

    # ---------------------------------------------------------------------------

    # compute the Harris corner response function for each point in the image
    #   R(x, y) = det(M(x, y) - k * tr(M(x, y))^2
    # where
    #             [      I_x(x, y)^2        I_x(x, y) * I_y(x, y) ]
    #   M(x, y) = [ I_x(x, y) * I_y(x, y)        I_y(x, y)^2      ] * G
    #
    # with "* G" denoting convolution with a 2D Gaussian.
    #
    # inputs:
    # - image: a grayscale image
    # returns:
    # - R: transformation of the input image to reflect "cornerness"
    def compute_corner_response(self, image):

        # Derivatives in x and y directions
        I_x = sobel(image, 0)
        I_y = sobel(image, 1)

        # A, B, C from matrix M defined in paper
        A = gaussian_filter(I_x ** 2, self.gaussian_sigma)
        B = gaussian_filter(I_y ** 2, self.gaussian_sigma)
        C = gaussian_filter(I_x * I_y, self.gaussian_sigma)

        # det(M), tr(M) as defined in paper
        det = A * B - C ** 2
        trace = A + B
        R = det - self.harris_corner_k * trace ** 2
        return R
 
    # ---------------------------------------------------------------------------

    # find (x,y) pixel coordinates of maxima in a corner response map
    # inputs:
    # - R: Harris corner response map
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the corner
    #   response map, assumed to be integer coordinates
    def get_keypoints(self, R):
        w = self.maxfilter_window_size
        n = int((w - 1) / 2)
        K = self.max_num_features
        # R[R != maximum_filter{R]] = 0 ??
        # Get all local maxima in wxw neighborhood?
        maximas = []
        maxes = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                u = max(0, i - n)
                d = min(i + n + 1, R.shape[0] - 1)
                l = max(0, j - n)
                r = min(j + n + 1, R.shape[1] - 1)
                # maxima = np.unravel_index(R[u:d, l:r].argmax(), (d - u, r - l))
                local_max = R[u:d, l:r].max()
                if R[i, j] == local_max:
                    maximas.append([i, j])
                    maxes.append(local_max)

        # Get K largest maxes?
        maxes.sort(reverse=True)
        maxes = maxes[0:K]
        # Match maxima to K maxes
        points = []
        for m in maximas:
            # if point is a local max and > 0...
            if R[m[0], m[1]] in maxes and R[m[0], m[1]] > 0:
                points.append([m[1], m[0]])
            if len(points) > K:
                break
        return np.array(points).astype(int)[0:K]
