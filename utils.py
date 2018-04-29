import cv2
import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

def get_img(filename):
    '''
    Inputs:
    filename -- filename of an image

    Return:
    img -- an image in RGB format
    '''
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

def get_imgs(dir):
    '''
    Inputs:
    dir -- directory of images folder

    Return
    imgs -- list of images in RGB format
    titles -- titles for respective images
    '''
    files = os.listdir(dir)
    imgs, titles = [], []
    for f in files:
        filename = f'{dir}/{f}'
        name = f.split('.')[0]
        img = get_img(filename)
        imgs.append(img)
        aug_img = cv2.flip(img, 1)
        imgs.append(aug_img)
        titles.append(name)
        titles.append(name)
    return imgs, titles

def get_data(dir):
    '''
    Inputs:
    dir -- directory of images

    Returns:
    all images in the directory in array format
    '''
    data, dirs = [], []
    # obtain all subdirectories that contains images
    for f in os.listdir(dir):
        dirs.append(f'{dir}/{f}')

    # read and store images in the subdirectories
    for dir in dirs:
        imgs, _ = get_imgs(dir)
        data.extend(imgs)

    # return all images in np.array format
    return np.array(data)

def plot_imgs(imgs, titles=None, rows=1):
    '''
    Inputs:
    imgs -- list of images
    titels -- optional, titles of images
    rows -- number of rows to display images
    '''
    cols = ceil(len(imgs) // rows)
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        if titles != None:
            plt.title(titles[i])
        plt.axis('off')

    plt.show()
