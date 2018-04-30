# Vehicle Detection
In this project, the goal is to write a software pipeline to detect vehicles in a video.

## Environment

This repo includes a yml file to create the environment for this project and you can run the following command:

```
conda env create -f environment.yml
```


## Training Data

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Some example images for testing the pipeline on single frames are located in the [test_images](https://github.com/lipeng2/CarND-VehicleDetection-P5/tree/master/test_images) folder and intermediate results of the output from each stage of the pipeline is stored in the folder [output_images](https://github.com/lipeng2/CarND-VehicleDetection-P5/tree/master/output_images)

## Overview

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) and histogram of color features extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Train a linearn SVM classifier

* Implement a sliding window technique, and incorporates with the trained SVM classifier to search for vehicles in images

* Use heat maps to reduce false positive detections

* Implement a tracker class to keep tracking vehicles

* Run pipeline on a video stream


## Histogram of Oriented Gradients


The code for HOG and color features extractions is contained in [features_extraction.py](https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/features_extraction.py). 

First we need to read in the training images, below is an example. 
<p align="center">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/examples/car_not_car.png" width="400">
</p>

Then we extract the HOG and color features, using `Hog` function from skimage package, and generate a histogram of color channels of the given image. Example shown below.

<p align="center">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/Figure_1.png" width="240">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/hog.png" width="240"> 
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/color.png" width="240">
</p>

Performing the same features extraction on each of training data, and append them to create features data for training a linear SVM classifier. Before using the features data to trian our model, the features need to be normalized so that the model can be more robust. 

## Train Linearn SVM Classifier


The code for training linear SVM is contained in [svm.py](https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/svm.py). 

We supply the features data obtained from step one to train our linear SVM model, and based on our empirical statistics, we found that using YUV color space, orietation=11, pix_per_cell=16, and cell_per_block=2 yields the best result, which the accuracy is achieved around 98.6%. In addition, we also augment the training images simply by flipping them horizontally using `cv2.flip`, and the accuracy of the model is able to improve to 99%. 

## Sliding Windows


The implementation is contained in [window_search.py](https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/window_search.py).

To implement sliding windows, we just need to first define the starting and stopping positions, the desired size of windows to search for, and the window overlapping rate. To illustrate the idea, we can use the implemented function `slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0, 0))`, we can generate the following result.

<p align="center">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/slide_windows.jpg" width="500">
</p>

And then we extract the features using aforementioned feature extraction method from each window and feed it to the our trained classifier to see if the window contains vehicles. However, instead of extracting features from individual window, which can be computationally expensive, the HOG features are extracted for the entire image (or a selected area), and then the features are subsampled according to the size of the window. If the window is classified contained vehicles, then we save it to our result. Below is an example,


<p align="center">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/search_window.png" width="500">
</p>

We can perform the same procedures with different window size and overlapping rates to obtain a more robust result. Below is a result using xy_window of (64, 64), (96, 96) and (128,128) with overlapping rate of 25% in both x and y directions. 


<p align="center">
  <img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/car_windows.png" width="500">
</p>

In addition, we create a heat map which is made by adding "heat" to all pixels within the windows where a positive detection is reported by our classifier. And the "heat" value is calculated as followed `heat = 1 + confidence scores * constant` given there is a positive detection, or `(test_prediction == 1)`. The confidence score, obtained using `SVM.decision_function`, represents the confidence of our classifier in positive detections. The higher the confidence score for the detection, the more likely the detection is positive. Multiplying the confidence scores with an appropriate constant can significantly help us to distinguish false postives from the true positives. Lastly, we apply heat map threshold and use `scipy.ndimage.measurements.label` to integrate all overlapping detection windows into one robust bounding box for each vehicle in the image, shown below.

<img src="https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/output_images/car_with_heat.jpg">

## Tracker Class

The code for tracker class is contained in [Tracker_v2.py](https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/Tracker_v2.py).
```python
def __init__(self):
        '''
        num_frames -- number of frames storing
        heatmaps -- list of previous heatmaps
        output_frames -- list of previous output frames
        '''
        self.num_frames = 20
        self.heatmaps = deque([], maxlen=self.num_frames)
        self.outputs_frames = deque([], maxlen=self.num_frames)
```
The Tracker class is used to track the heatmaps from the previous 20 frames. It enables us to produce average heatmaps over 20 frames, which can further eliminate false postives that only appear in a few frames while leviating issue of wobbly detections. Additionally, the Tracker class memorizes the 20 most recent average heatmaps in order to enhance the robustness of our pipeline. 


## Improvement


* One of the biggest problem with the pipeline is computaional time. Features extraction is extremely time consuming, and it will be difficult to deploy this pipeline to perform real-time vehicles detection. One of the solution can be using Deep learning model, where features extraction is not required. One example will be to use [YOLO](https://pjreddie.com/media/files/papers/yolo.pdf).

* If not given enough data to train a deep neural network, we can still use SVM and features extraction. Once we have identified the vehicles in the video, we can devise a model to project where they are going to be in the next few frames based on their speed and the moving directions, and then we slide the windows accordingly. By doing so, we can reduce the number of times of doing features extractions and hypothetically speed up the pipeline performances several times. 
