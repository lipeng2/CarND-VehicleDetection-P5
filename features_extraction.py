import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from utils import *

def bin_spatial(img, size=(32, 32)):
    '''
    Inputs:
    img -- an image in RGB format
    size -- resize parameter

    Return:
    a one dimensional feature vector
    '''

    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Inputs:
    img -- an image in RGB format
    nbins -- number of bins of histogram
    bins_range -- range of each bin of histogram

    Return:
    rhist -- color histogram of red channel
    ghist -- color histogram of green channel
    bhist -- color histogram of blue channel
    bin_centers -- centers of bins
    hist_features -- histogram features
    '''
    # compute the histogram of the RGB channels
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def single_hog_features(img, orient=11, pix_per_cell=16, cell_per_block=2, vis=False, feature_vec=True):
    '''
    Inputs:
    img -- an image in RGB format
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized

    Return
    features -- hog featuers
    '''

    if vis == True:
        features, hog_img = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")

        return features, hog_img

    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")

        return features

def all_hog_features(img, orient=11, pix_per_cell=16, cell_per_block=2):
    '''
    Inputs:
    img -- an image in RGB format
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized

    Return
    hog featuers for all channels of the given image 
    '''
    ch1, ch2, ch3 = img[:,:,0], img[:,:,1], img[:,:,2]
    hog1 = single_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = single_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = single_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    return hog1, hog2, hog3

def get_hog_features(img, orient=11, pix_per_cell=16, cell_per_block=2):
    hog_features = []
    for channel in range(img.shape[2]):
        hog_features.append(single_hog_features(img[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    return hog_features

def single_extract_features(img, cspace='RGB', spatial_size=(32,32), hist_bins=32, orient=11, pix_per_cell=16, cell_per_block=2):
    '''
    Inputs:
    img -- image in RGB format
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized

    Return
    features -- hog featuers
    '''
    if cspace != 'RGB':
        cvt='cv2.COLOR_RGB2'+cspace
        img = cv2.cvtColor(img, eval(cvt))
    img_features = []
    img_features.append(bin_spatial(img, spatial_size))
    img_features.append(color_hist(img, hist_bins))
    img_features.append(get_hog_features(img, orient, pix_per_cell, cell_per_block))

    return np.concatenate(img_features)

def extract_features(imgs, cspace='RGB', spatial_size=(32,32), hist_bins=32, orient=11, pix_per_cell=16, cell_per_block=2):
    '''
    Inputs:
    imgs -- images in RGB format
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized

    Return
    features -- hog featuers
    '''
    features = []
    for img in imgs:
        features.append(single_extract_features(img, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block))

    return features
