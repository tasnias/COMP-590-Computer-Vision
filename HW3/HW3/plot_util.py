# Tasnia Sarwar
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import matplotlib.pyplot as plt
import numpy as np

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip


#-------------------------------------------------------------------------------

# display keypoints in an image
#
# inputs:
# - image: input image, assumed to be grayscale
# - keypoints: Nx2 array of keypoint (x,y) pixel locations in the image
# returns: None
def plot_keypoints(image, keypoints):
    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:,0], keypoints[:,1], s=8, c="r")
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)


#-------------------------------------------------------------------------------

# display keypoint matches between a pair of images
#
# inputs:
# - image1: first input image, assumed to be grayscale
# - keypoints1: N1 x 2 array of keypoint (x,y) pixel locations in the first
#   image, assumed to be integer coordinates
# - image2: second input image, assumed to be grayscale
# - keypoints2: N2 x 2 array of keypoint (x,y) pixel locations in the second
#   image, assumed to be integer coordinates
# - matches: M x 2 array of indices for the matches; the first column
#   provides the index for the keypoint in the first image, and the second
#   column provides the corresponding keypoint index in the second image
# returns: None
def plot_matches(image1, keypoints1, image2, keypoints2, matches):
    keypoints1 = keypoints1[matches[:, 0]]
    keypoints2 = keypoints2[matches[:, 1]]

    # show the original images side-by-side
    im = np.column_stack((image1, image2))
    plt.imshow(im, cmap="gray")
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1],
		  facecolors='none', edgecolors='r')
    plt.scatter(keypoints2[:, 0] + image1.shape[1], keypoints2[:, 1],
		  facecolors='none', edgecolors='r')

    # draw the matches with random colors
    for kp1, kp2 in izip(keypoints1, keypoints2):
        x = (kp1[0], kp2[0] + image1.shape[1])
        y = (kp1[1], kp2[1])
        c = (2. * np.random.rand(3) + 1.) / 3. # RGB \in (0.5, 1)
        plt.plot(x, y, color=c, linewidth=1.0)

    plt.gca().axis('off')
    plt.tight_layout()
