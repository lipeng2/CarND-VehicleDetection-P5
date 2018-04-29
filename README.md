# Vehicle Detection
In this project, the goal is to write a software pipeline to detect vehicles in a video.

## Environment

This repo includes a yml file to create the environment for this project and you can run the following command:

```
conda env create -f environment.yml
```

## Overview

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) and histogram of color features extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Implement a sliding window technique, and incorporates with trained SVM classifier to search for vehicles in images

* Implement a tracker class to eliminate false positve detections and keep tracking vehicles

* Run pipeline on a video stream

### Training Data
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Some example images for testing the pipeline on single frames are located in the [test_images](https://github.com/lipeng2/CarND-VehicleDetection-P5/tree/master/test_images) folder and intermediate results of the output from each stage of the pipeline is stored in the folder [output_images](https://github.com/lipeng2/CarND-VehicleDetection-P5/tree/master/output_images)

### Histogram of Oriented Gradients

The code for HOG and color features extractions is contained in [features_extraction.py](https://github.com/lipeng2/CarND-VehicleDetection-P5/blob/master/features_extraction.py).


