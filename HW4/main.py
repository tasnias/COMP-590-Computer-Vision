# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: Panorama

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

from harris.feature_matcher import FeatureMatcher
from harris.harris_corner import HarrisCornerFeatureDetector
from harris.plot_util import plot_keypoints, plot_matches
from h_matrix_ransac import ransac

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)


# -------------------------------------------------------------------------------

# convert a color image to a grayscale image according to luminance
# this uses the conversion for modern CRT phosphors suggested by Poynton
# see: http://poynton.ca/PDFs/ColorFAQ.pdf
#
# input:
# - image: RGB uint8 image (values 0 to 255)
# returns:
# - gray_image: grayscale version of the input RGB image, with floating point
#   values between 0 and 1
def rgb2gray(image):
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (0.2125 * red + 0.7154 * green + 0.0721 * blue) / 255.

def bilinearInterpolation(x, y, image, offset):
    if x - offset[0] < 0 or y - offset[1] < 0 or math.ceil(x) - offset[0] >= image.shape[1] or math.ceil(y) - offset[1] >= image.shape[0]:
        return 0
    else:
        # x = x - offset[0]
        # y = y - offset[1]
        d1 = x - math.floor(x)
        d2 = 1 - d1
        d3 = y - math.floor(y)
        d4 = 1 - d3
        v1 = image[math.floor(y), math.floor(x)]
        v2 = image[math.floor(y), min(math.floor(x) + 1, image.shape[1])]
        v3 = image[min(math.floor(y) + 1, image.shape[0]), math.floor(x)]
        v4 = image[min(math.floor(y) + 1, image.shape[0]), min(math.floor(x) + 1, image.shape[1])]
        a4 = d1 * d3 / ((d1 + d2) * (d3 + d4))
        a3 = d2 * d3 / ((d1 + d2) * (d3 + d4))
        a2 = d1 * d4 / ((d1 + d2) * (d3 + d4))
        a1 = d2 * d4 / ((d1 + d2) * (d3 + d4))
        q = v1 * a1 + v2 * a2 + v3 * a3 + v4 * a4
        return q

def warpPerspective(image, H, shape, offset):
    result = np.zeros((shape[1], shape[0]), dtype='float32')
    Hinv = np.linalg.inv(H)
    for point in np.ndenumerate(image):
        y, x = point[0]
        x_, y_, s = Hinv @ [x, y, 1]
        x_, y_ = x_ / s, y_ / s
        result[y,x] = bilinearInterpolation(x_, y_, image, offset)
    return result


def stitchImages(image1, image2, H):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    bounds1 = np.float32([[0, 0, 1], [0, height1, 1], [width1, height1, 1], [width1, 0, 1]])
    bounds2 = np.float32([[0, 0, 1], [0, height2, 1], [width2, height2, 1], [width2, 0, 1]])
    bounds1_warped = np.int32([H @ pt for pt in bounds1])
    bounds1_warped = np.int32([pt / pt[2] if pt[2] != 0 else pt for pt in bounds1_warped])
    pts = np.concatenate((bounds2, bounds1_warped))
    mins = np.int32([np.min(pts[:, 0]), np.min(pts[:, 1])])
    maxes = np.int32([np.max(pts[:, 0]), np.max(pts[:, 1])])
    T = np.array([[1, 0, -mins[0]], [0, 1, -mins[1]], [0, 0, 1]])
    # Couldn't get my own bilinear interpolation to work so using OpenCV's implementation
    # result = warpPerspective(image1, T@H, tuple(maxes - mins), mins)
    result = cv2.warpPerspective(image1, T @ H, tuple(maxes-mins))
    result[-mins[1]:height2 + -mins[1], -mins[0]:width2 + -mins[0]] = image2
    return result


def main(args):
    # create the feature extractor
    extractor = HarrisCornerFeatureDetector(args)
    images = []
    for image in args.images.split(","):
        images.append(rgb2gray(plt.imread(image)))

    base = images[0]
    for i in range(1, len(images)):
        image = images[i]
        # keypoints: Nx2 array of (x, y) pixel coordinates for detected features
        keypoints1 = extractor(base)
        keypoints2 = extractor(image)

        # create the feature matcher and perform matching
        feature_matcher = FeatureMatcher(args)
        matches = feature_matcher(base, keypoints1, image, keypoints2)

        # compute homography and get inlier matching
        data = np.column_stack((keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]))
        H, inlier_mask = ransac(data, args.inlier_threshold, args.confidence_threshold, args.max_num_trials)
        base = stitchImages(image, base, np.linalg.inv(H))

    # display the keypoints
    plt.figure(1)
    plot_keypoints(base, keypoints1)

    plt.figure(2)
    plot_keypoints(image, keypoints2)

    # display the matches between the images
    plt.figure(3)
    plot_matches(base, keypoints1, image, keypoints2, matches)
    #
    # display the matches after RANSAC
    plt.figure(4)
    plot_matches(base, keypoints1, image, keypoints2, matches[inlier_mask])

    plt.figure(5)
    plt.imshow(base, cmap='gray')
    plt.show()
    plt.imsave('stitched.png', base, cmap='gray')

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and display Harris Corner features for "
                    "images, and match the features between the images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("images", type=str, help="list of input images, separated by commas")

    # Harris corner detection options
    parser.add_argument("--harris_corner_k", type=float, default=0.05,
                        help="k-value for Harris' corner response score")
    parser.add_argument("--gaussian_sigma", type=float, default=1.,
                        help="width of the Gaussian used when computing the corner response; usually set to a value between 1 and 4")
    parser.add_argument("--maxfilter_window_size", type=int, default=11,
                        help="size of the (square) maximum filter to use when finding corner maxima")
    parser.add_argument("--max_num_features", type=int, default=1000,
                        help="(optional) maximum number of features to extract")
    parser.add_argument("--response_threshold", type=float, default=0.01,
                        help="when extracting feature points, discard any points whose corner response is less than this value times the maximum response")

    # feature description and matching options
    parser.add_argument("--matching_method", type=str, default="ssd",
                        choices=set(("ssd", "ncc")),
                        help="descriptor distance metric to use")
    parser.add_argument("--matching_window_size", type=int, default=7,
                        help="window size (width and height) to use when matching; must be an odd number")

    parser.add_argument("--inlier_threshold", type=float, default=100.,
                        help="point-to-line distance threshold, in pixels, to use for RANSAC")

    parser.add_argument("--confidence_threshold", type=float, default=0.99,
                        help="stop RANSAC when the probability that a correct model has been  found reaches this threshold")
    parser.add_argument("--max_num_trials", type=float, default=50000,
                        help="maximum number of RANSAC iterations to allow")

    args = parser.parse_args()
    main(args)
