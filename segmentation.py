#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imread


def norm_hist(ima):
    histogram, bin_edges = np.histogram(
        ima.flatten(), bins=256, range=(15, 256)
    )
    return 1. * histogram / np.sum(histogram)


def display_hist(ima, vmin=None, vmax=None):
    plt.figure(figsize=[10, 5])
    nh = norm_hist(ima)
    plt.subplot(1, 2, 1)
    plt.imshow(ima, cmap=cm.gray, vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    plt.plot(nh, label='hist.')
    plt.legend()
    plt.xlabel('gray level')
    plt.show()


def _score(img):
    detected = np.nonzero(img)
    # every pixel corresponds to 0.013225 cm² of tumor area
    score = 0.013225 * np.size(detected)
    print("The measured area of the tumor is of approximately %.2f cm²" % score)


mri = img_as_ubyte(imread(str(sys.argv[1]), as_gray=True))
mode = str(sys.argv[2])
display_hist(mri)

# apply basic segmentation based on gray level
# then refine the result with morphological transforms
# very fast operation, is good for approximating
if mode == "seg_morph":
    seg = mri > 175  # threshold value based on observation
    seg = cv.morphologyEx(seg.astype('uint8'), cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # close areas
    seg = cv.morphologyEx(seg.astype('uint8'), cv.MORPH_OPEN, np.ones((13, 13), np.uint8))  # remove small/non-desired
    # shows the original MRI with the detected tumour highlighted on it
    plt.figure()
    plt.imshow(mri, cmap=cm.gray)
    plt.imshow(seg, alpha=0.3)
    plt.savefig('mri_segMorph.png')
    _score(seg)


else:
    print("watershed mode")
