# **Behavioral Cloning** 

The goal of this project is to develope a deep neural network that can learn the steering control to drive the car on a track. Network is trained over the driving training data collection from a simulator.

Following steps are required to accomplish the goal:
* By the use of simulator, collection the training data for autonomous driving.
* Build a deep neural network that can predict the steering angles.
* Train and validate the model with a training and validation data collected.
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/1_centre.jpg "Centre Camera"
[image3]: ./examples/1_left.jpg "Left Camera"
[image4]: ./examples/1_right.jpg "Right Camera"
[image5]: ./examples/2_centre.jpg "Recovery Driving"


### Code Files

Project includes the following files:
* model.ipynb:       script to create and train the network.
* drive.py:          to drive the car in autonomous mode by using trained model
* model.h5:          trained neural network to drive the car in autonomous mode.
* video.mp4          autonomous driving video by the network   
* writeup_report.md: detailed summary of project and network architecture.

#### 2. Autonomous Driving using Trained Model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 [record_dir]
```

Optional recorded directory stores frame images of the autonomous driving which can be used to make the video of autonomous driving by executing 
```sh
python video.py <recorded_dir>
```


### Model Architecture and Training Strategy

#### Architecture

Recorded training data is set of images of the road track. A neural network to be able to learn steering control of the vehicle, must be 
able to recognize these road tracks. Therefore, I have used a convolutional deep neural network for this task.

A convolution of sizes 16, 32, and 64 are used with filter of 4x4. Each convolution is followed by max pooling of 2x2 kernal. Each layer of the convolution follows the batch normalization to accelerate batch training.

Below diagram shows the complete network architecture, also included in the python notebook (model.ipynb).

![alt text][image1]

#### Attempts to reduce overfitting in the model

The model contains batch normalization for the convolutional later and dropout for the fully connected layers to reduce any overfitting of the training data.

Complete training data has been splitted into train and validation set. The model is trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Data has been normalized with range [-1, 1]. To reduce the noise in training, recorded images are cropped to remove unwanted information from images.

#### 3. Model parameter tuning

The model is using an adam optimizer, so the learning rate was not tuned manually. Model is trained over the batch size of 64 images, with total 94,272 training images. Network has been evaluated on total 23,552 validation samples.


#### 4. Training data

Training data was collected by driving the vehicle on road in simulator. To generalize the training data set, following data augumentation techniques were used: 

##### Using Centre, Left and Right Cameras
There are three cameras mounted on the car - centre, left and right. During the training, steering measurements are recorded with respect to centre camera image only. Training data records all camera images of the time frame. To measure steering control with respect to left and right camera images, a correction factor of 0.23 has been applied to centre camera steering measurement. Inclusion of left and right camera images helps in generalizing the training of the network.

![alt text][image2] ![alt text][image3] ![alt text][image4]

##### Flipped Images
Each camera image has been flipped by mirror image and corresponding angle has been adjusted in reverse direction. This gives better generalization of the track and covers the case of driving in the track in opposite direction in exact conditions. 

##### Off-Track Driving
Off track driving was recorded to learn how to bring the car back to track.

![alt text][image5]

##### Opposite Direction Driving
Training data was also collected by driving the car in opposite direction of the track.

##### Repeated Driving on Hard/ Curved Tracks
Several driving attempts were recorded to cover sharp curved driving.

#### 5. Results
Trained model can successfully drive the car by keeping the car in the centre of the road. Sharp, curves turns are handled well and car does not go beyond the track. 

video.mp4 has been recorded to show autonomous driving.