#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imread


# return the normalized np array of the histogram
def norm_hist(ima):
    histogram, bin_edges = np.histogram(
        # histogram range changed to remove 'garbage' low light values
        ima.flatten(), bins=256, range=(25, 256)
    )
    return 1. * histogram / np.sum(histogram)


# find optimal threshold in an histogram h from an initial threshold t
# (taken from Module 3 correction)
def optimal_threshold(h, t):
    # Cut distribution in 2
    g1 = h[:t]
    g2 = h[t:]
    # Compute the centroids
    m1 = (g1 * np.arange(0, t)).sum() / g1.sum()
    m2 = (g2 * np.arange(t, len(h))).sum() / g2.sum()
    # Compute the new threshold
    t2 = int((m1 + m2) / 2)
    if t2 != t:
        return optimal_threshold(h, t2)
    return t2


# print on the standard output the measured tumor area
def _score(img):
    detected = np.nonzero(img)
    # every pixel corresponds to 0.013225 cm² of tumor area
    score = 0.013225 * np.size(detected)
    print("The measured area of the tumor is approximately of %.2f cm²" % score)


mri = img_as_ubyte(imread(str(sys.argv[1]), as_gray=True))

# a basic gray value segmentation is made based on the optimal threshold
t = optimal_threshold(norm_hist(mri), 150)
seg = mri > t + 25  # threshold value is shifted to take into account initial low value removal
# morphological transforms are used to refine the result
seg = cv.morphologyEx(seg.astype('uint8'), cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # close areas
seg = cv.morphologyEx(seg.astype('uint8'), cv.MORPH_OPEN, np.ones((13, 13), np.uint8))  # remove small/non-desired
# shows the original MRI with the detected tumour highlighted on it
plt.figure()
plt.imshow(mri, cmap=cm.gray)
plt.imshow(seg, alpha=0.3)
plt.savefig('mri_segMorph.png')
_score(seg)
