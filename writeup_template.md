# **Behavioral Cloning**


**Project Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_image.png "Example of Center, left and Right Images"
[image2]: ./writeup_images/steering_angle_histogram.png "Distribution of Steering Angles"
[image3]: ./writeup_images/model_summary.png "Summary of Keras Model"
[image4]: ./writeup_images/mse_loss.png "Model loss"
[image5]: ./writeup_images/flipped_images.png "Example of Flipped Center, Left and Right Images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py**: Containing the script to create and train the model.
* **drive.py**: For driving the car in autonomous mode (this file is provided in the project, I did not modify it).
* **model.h5**: Containing a trained convolution neural neural.
* **run1.mp4**: A video of 1 autonomously driven round on track 1.
* **writeup_report.md**: This file summarising the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
This is illustrated in the run1.mp4 video. I trained and tested the model only using track 1.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.

The images for training and validation are loaded and prepared using the 2 functions readCSVFile() and makeListOfPathsAndMeasurements() in lines 13-51.

The model itself is defined in the function nVidiaModel() in lines 55-76.

The generator for the model fitting is define in the function generator() in lines 78-109.

The loading of data, compilation and training of the model is found in lines 112-141, the last line saving the model.h5 file.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I went directly for the nVidia neural network which was introduced in the project videos. My implementation of this CNN looks like this:

![alt text][image3]

I did not modify the model by adding dropout or pooling.

I used a Lambda layer for normalising the images, and the Keras Cropping2D function for remove the upper 50 and the lower 20 lines of the input images.

#### 2. Attempts to reduce overfitting in the model

The collected data was divided into 80% training data and 20% validation data using the standard train_test_split() function in line 121 of model.py.

After experimentation with the model and the training/validation data I found that overfitting started to happen after 3 training epochs. Instead of adding measures to reduce the overfitting I chose to stop the training after 3 epochs and that proved good enough to make the trained network drive the car around track 1.

For timing reasons I decided to not experiment further, but it would certainly be interesting to for example add some dropout layers and then train the network for more epochs, and see its performance.

#### 3. Model parameter tuning

The model used the adam optimizer, and the learning rate was not tuned manually (model.py line 132).

#### 4. Appropriate training data

First I used the training data provided by Udacity. That produced a network that almost worked. I then collected my own data, in 2 sessions. In total my collected consists of 5 rounds of driving in the center of the road and 2 rounds of recovery driving, where I let the car drift to the sides of the road before steering it back to the center.

Instead of driving the track in the opposite direction I flipped all the images in my generator function (lines 104-105). I used the openCV flip function and multiplied the steering angle with -1.0, as suggested in the project lesson material. The image flipping ensures that the training data is not biased to steer in one particular direction.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Having carefully following the project lesson material I chose to directly implement the nVidia Autonomous Car Group Model. It seemed to be the right to do, and the results proved this decision to be correct.

As mentioned above I split my collected data in 80% training and 20% validation data. I collected data 2 times, as my first collected data set didn't produce a working model. I decided to collect more data.

I used all collected images; i.e. center, left and right images. Examples of my training images are:

![alt text][image1]

I chose to adjust the collected steering angle with +0.25 for the left image and -0.25 for the right image. This is implemented in lines 46 and 47 of model.py.

I chose to flip the training and validation data in my generator function, which created twice as much data, and ensured the training data was not biased to steer in a certain direction. The same images as above but flipped look like this:

![alt text][image5]

Finally I normalized the image data to be between 0.5 and -0.5, and I cropped the upper 50 lines and the lower 20 lines off each image, to remove the landscape data and the hood of the car, and create images with only the relevant part, i.e. the road.

I did not change the images to gray scale. Certainly it would be worthwhile to experiment with this.

It proved important to change the image format to RGB before training the network. cv2.imread returns BGR images and using those directly does not produce the required results. It is quite obvious when you think about it. The BGR images look differently that the RGB images the network will be feed from the car in autonomous driving mode. Changing from BRG to RGB is done in model.py line 96 and 97.

The following histogram shows the distribution of my 32013 collected steering angles **before** image flipping. As it can be seen the data is already reasonably distributed. What can also be seen is as expected that most steering angles are in the middle, and quite few at the extremes.

![alt text][image2]

For a better trained model obviously this distribution need addressing, in short we need more training data with larger steering angles.

As mentioned I had in total 32013 images. Flipping them off course doubled this figure, giving me in the end 51220 training images and 12806 validation images.

I stopped the model training after 3 epochs, as I observed overfitting starting to occur with more epochs. The trained model outputted the following loss:

![alt text][image4]

With this model the vehicle is able to drive autonomously around the track without leaving the road, as you can see in the submitted run1.mp4 file.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-76) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image3]

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, in total I 5 laps on track one using center lane driving, and 2 laps of the vehicle recovering from the left side and right sides of the road back to center.

I did not use track 2 as the collected proved good enough to create a model which can drive the car around track 1. I am a little short of time due to working commitments, so I stopped the work when I had achieved the goals of the project. Using data from track 2 would certainly be interesting.

As mentioned above I doubled my training data set by flipping the images, and I applied normalisation and cropping the upper and lower image parts before applying to the training of the model.

After this process, I had 51220 training images and 12806 validation images.

My generator function (line 78-109 in model.py) randomly shuffles the data sets.
