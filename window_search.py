import numpy as np
import cv2
from features_extraction import *
from scipy.ndimage.measurements import label

def draw_boxes(img, boxes, color=(0,255,0), thickness=3):
    '''
    Inputs:
    img -- an image
    boxes -- list of tuples of coordinates defining boxes
    color -- RGB values
    thickness -- thickness of boxes

    Return
    draw_img -- boxes drawn on the give image
    '''
    draw_img = img.copy()
    for box in boxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thickness)
    return draw_img

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Inputs:
    img -- an image in RGB format
    x_start_stop -- starting and stopping of x positions
    y_start_stop -- starting and stopping of y positions
    xy_window -- size of sliding window
    xy_overlap -- percentage of overlaping of sliding windows

    Return:
    window_list -- a list of all sliding windowns
    '''
    # If x and/or y start/stop positions not defined, set to image size
    xs, xp = x_start_stop
    ys, yp = y_start_stop
    xs = 0 if xs == None else xs
    ys = 0 if ys == None else ys
    xp = img.shape[1] if xp == None else xp
    yp = img.shape[0] if yp == None else yp
    # Compute the span of the region to be searched
    xspan, yspan = xp-xs, yp-ys
    # Compute the number of pixels per step in x/y
    xstep_pix, ystep_pix = int(xy_window[0]*(1-xy_overlap[0])), int(xy_window[0]*(1-xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = xspan // xstep_pix - 1
    ny_windows = yspan // ystep_pix - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for y in range(int(ny_windows)):
        for x in range(int(nx_windows)):
            xstart = x*xstep_pix + xs
            xend = xstart+xy_window[0]
            ystart = y*ystep_pix + ys
            yend = ystart+xy_window[1]
            window_list.append(((xstart, ystart), (xend, yend)))

    return window_list

def search_windows(img, windows, clf, scaler, cspace='RGB', spatial_size=(32,32), hist_bins=32, orient=11, pix_per_cell=16, cell_per_block=2):
    '''
    Inputs:
    img -- image in RGB format
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized

    Return
    on_windows -- a list of windows that are predicted containing vehicles
    '''

    # crate a list to store windows that contain vehicles
    on_windows = []

    for window in windows:
        # get window portion of the image
        window_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        # resize the window image
        test_img = cv2.resize(window_img, (64,64))
        # extract features
        features = single_extract_features(test_img, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block)
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # make prediction
        pred = clf.predict(test_features)

        if pred == 1:
            on_windows.append(window)

    return on_windows

def get_heatmap(img, bbox_list, confids, threshold):
    '''
    img -- an image in RGB format
    bbox_list -- windows
    confids -- confidence scores on windows to indicating whether it contains a vehicle
    threshold -- heatmap threshold

    Return:
    heatmap -- a heatmap indicates areas with probability where vehicles reside
    '''
    heatmap = np.zeros_like(img[:,:,0])
    for i, box in enumerate(bbox_list):
        value =  int(1+confids[i]*5)
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # apply threshold
    heatmap[heatmap <= threshold] = 0

    return heatmap

def get_labeled_bboxes(img, labels, draw=True):
    '''
    Inputs:
    img -- an image in RGB foramt
    labels -- labels of vehicles

    Return:
    img -- an image with bounding boxes identifying positions of vehicles
    windows -- bounding boxes positions
    '''
    # Iterate through all detected cars
    windows = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        windows.append((bbox[0], bbox[1]))
        if draw:
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 3)
    if draw:
        return img, windows
    else:
        return windows
        
def find_cars(img, ystart, ystop, scales, svc, X_scaler, cells_per_step=1, orient=11, pix_per_cell=16, cell_per_block=2, spatial_size=(32,32), hist_bins=32):
    '''
    Inputs:
    img -- an image in RGB format
    x_start_stop -- starting and stopping of x positions
    y_start_stop -- starting and stopping of y positions
    scales -- a list of different scalings
    svc -- vehicle classifier
    X_scaler -- scaler for normalization transformation
    orient -- number of orientation bins that the gradient infomation will be split up into in the histogram
    pixels_per_cell -- cell size over which each gradient histogram is computed
    cell_per_block -- size of local area over which the histogram counts in a given cell will be normalized
    spatial_size -- size of color spatial bins
    hist_bins -- size of color histogram bins

    Return:
    window_list -- a list of all sliding windowns
    '''

    windows, confids = [], []
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)

    for scale in scales:
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Compute individual channel HOG features for the entire image
        hog1, hog2, hog3 = all_hog_features(ctrans_tosearch, orient, pix_per_cell, cell_per_block)

        # Define blocks and steps
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    confids.append(svc.decision_function(test_features))

    return windows, confids

def region_interet(heatmap, hot_windows):
    for box in hot_windows:
        max = np.max(np.max(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] = max
    return heatmap

if __name__ == "__main__":
    from utils import *
    from sklearn.preprocessing import StandardScaler
    from sklearn.externals import joblib
    import pickle
    from scipy.ndimage.measurements import label
    from time import time


    images, titles = get_imgs('test_images')
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('model.pkl')

    for img in images:

        windows, confids = find_cars(img, ystart=400, ystop=620, scales=np.arange(1,2,0.5), svc=clf, X_scaler=scaler, cells_per_step=1, orient=11, pix_per_cell=16, cell_per_block=2, spatial_size=(32,32), hist_bins=32)
        draw_img = draw_boxes(img, windows)
        heatmap= get_heatmap(img, windows, confids, 4)
        _, hot_windows = get_labeled_bboxes(np.copy(img), label(heatmap))

        new = np.zeros_like(heatmap)
        for box in hot_windows:
            ave = int(np.mean(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]))
            print(ave, type(ave))
            new[box[0][1]:box[1][1], box[0][0]:box[1][0]] = ave

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
