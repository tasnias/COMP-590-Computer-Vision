# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import matplotlib.pyplot as plt
import numpy as np

from feature_matcher import FeatureMatcher
from harris_corner import HarrisCornerFeatureDetector
from plot_util import plot_keypoints, plot_matches

#-------------------------------------------------------------------------------

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
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    return (0.2125 * red + 0.7154 * green + 0.0721 * blue) / 255.

#-------------------------------------------------------------------------------

def main(args):
    # create the feature extractor
    extractor = HarrisCornerFeatureDetector(args)

    # convert uint8 RGB images to grayscale images in the range [0, 1]
    image1 = plt.imread(args.image1)
    image1 = rgb2gray(image1)

    image2 = plt.imread(args.image2)
    image2 = rgb2gray(image2)

    # keypoints: Nx2 array of (x, y) pixel coordinates for detected features
    keypoints1 = extractor(image1)
    keypoints2 = extractor(image2)

    # create the feature matcher and perform matching
    feature_matcher = FeatureMatcher(args)
    matches = feature_matcher(image1, keypoints1, image2, keypoints2)

    # display the keypoints
    plt.figure(1)
    plot_keypoints(image1, keypoints1)

    plt.figure(2)
    plot_keypoints(image2, keypoints2)

    # display the matches between the images
    plt.figure(3)
    plot_matches(image1, keypoints1, image2, keypoints2, matches)

    plt.show()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and display Harris Corner features for two "
            "images, and match the features between the images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("image1", type=str, help="first input image")
    parser.add_argument("image2", type=str, help="second input image")

    # Harris corner detection options
    parser.add_argument("--harris_corner_k", type=float, default=0.05,
        help="k-value for Harris' corner response score")
    parser.add_argument("--gaussian_sigma", type=float, default=1.,
        help="width of the Gaussian used when computing the corner response; "
             "usually set to a value between 1 and 4")
    parser.add_argument("--maxfilter_window_size", type=int, default=11,
        help="size of the (square) maximum filter to use when finding corner "
             "maxima")
    parser.add_argument("--max_num_features", type=int, default=1000,
        help="(optional) maximum number of features to extract")

    # feature description and matching options
    parser.add_argument("--matching_method", type=str, default="ssd",
        choices=set(("ssd", "ncc")),
        help="descriptor distance metric to use")
    parser.add_argument("--matching_window_size", type=int, default=7,
        help="window size (width and height) to use when matching; must be an "
             "odd number")

    args = parser.parse_args()

    main(args)
