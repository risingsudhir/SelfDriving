# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/train_image_colour.png "ColourChannel"
[image3]: ./examples/train_image_gray.png "Grayscaling"
[image4]: ./examples/feature_map_1.png "Feature Map"
[image5]: ./examples/Image_1.jpg "Traffic Sign 1"
[image6]: ./examples/Image_2.jpg "Traffic Sign 2"
[image7]: ./examples/Image_3.jpg "Traffic Sign 3"
[image8]: ./examples/Image_4.jpg "Traffic Sign 4"
[image9]: ./examples/Image_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Next sections provide descritpion of the project, underlying model, training and test results. Project can be accessed from github from this [link](https://github.com/risingsudhir/SelfDriving/blob/master/TrafficSignClassification/dlnd_image_classification.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Pandas library has been used to analyse the traffic sign data. Below are statistics of the data:

* The size of training set is 32,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Chart below shows the distribution of sign classes in the trainig data. This distribution highlights the uneven population of traffic signs in the training set. This problem can be trackled by imputing the training data with fake images.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Preproceesing pipeline is built with following steps: 

1- Converting images from RGB to Gray scale. This helps in reducing network's sensitivity to the background the color noise of the image.

2- Normalized the image data in the range of [-1, 1] for better convergance of gradient descent.

![alt text][image2] ![alt text][image3]

As shown in the histogram, traffic sign classes do not have equal population in the training set. This may lead to network bias on high density classes. To avoid this, we can generate fake data for clases having small sample sample sizes. For this project, I have not augumented the training data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

A typical convolution nnetwor architecure has been used to identify traffic signs. Below is the complete network architectue:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 1x4   	| 1x1 stride, valid padding, outputs 28x28x4 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x4 				    |
| Convolution 4x16	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 16x64	    | 1x1 stride, same padding, outputs 5x5x64  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x64 				    |
| Fully connected		| 576x1024   									|
| RELU					|												|
| Fully connected		| 1024x256     									|
| RELU					|												|
| Fully connected		| 256x64     									|
| RELU					|												|
| Fully connected		| 64x43     									|
| Softmax				| 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I have used Adaptive optimizer with softmax cross entropy for loss function. Model is trained with dropout probability of 75%. Batch size of 256 and a learning rate of 0.0001 was choosen based on network convergence and loss.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy:   99.99%
* validation set accuracy: 95.32% 
* test set accuracy:       95.33%

Although my model is standard convolution network model, I have chosen a different architecture than LeNet architectur. I have chosen this model based on iterative approach to find the best model based on performance on training, validation and test set.

First 3 layers of convolution are used to build the deep feature maps without being too aggressive on the gray scale image which has just one channel for depth.

Fully connected layers are enabled with dropout to avoid overfitting the training data. Model achieves training accuracy of 99.99%, validation accuracy of 95.32 and test accuracy of 95.33% which shows training and test accuracy are comparable and model is not an overfit or underfit of the current training data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I have used to test the model, which are not part of of the traing, validation and test set data.  

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

These images were destorted slightly to check the sensitivity of the model. Image #2 (30 km speed limit) has close focus and image #4 differs significantly from standrd children crossing image with distorted resolution.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)                  | Speed limit (70km/h)                        | 
| Road work                             | Road work                                   |
| Speed limit (30km/h)                  | No passing for vehicles over 3.5 metric tons|
| Children crossing                     | Road work                                   |
| Right-of-way at the next intersection | Right-of-way at the next intersection       |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. With chosen complexity of Image #2 and Image #4, this compares favorably to the accuracy on the test set of 95.33%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Section #3 of the python notebook performs prediction on completely new set of images. Section prints top 5 predictions with probabilities that network is thinking about the image.

IMAGE #1: Speed limit (70km/h) 
![alt text][image5]

| Probability   |     Prediction                       | 
|:-------------:|:------------------------------------:| 
| 0.992010      | Speed limit (70km/h)                 | 
| 0.006608      | Speed limit (60km/h)                 |
| 0.001341      | Speed limit (20km/h)                 |
| 3.72503e-05   | Bumpy Road                           |
| 3.74297e-06   | Slippery Road                        |


IMAGE #2: Road work  
![alt text][image6]

| Probability   |     Prediction                       | 
|:-------------:|:------------------------------------:| 
| 1.0           | Road work                            | 
| 3.56645e-33   | Wild animals crossing                |
| 1.13135e-35   | Ahead only                           |
|  6.79941e-36   | Bumpy road                           |
| 2.20148e-36   | Double curve                         |


IMAGE #3: Speed limit (30km/h) 
![alt text][image7]

| Probability   |     Prediction                               | 
|:-------------:|:--------------------------------------------:| 
| 0.999617      | No passing for vehicles over 3.5 metric tons | 
| 0.00033141    | Speed limit (30km/h)                         |
| 2.61555e-05   | Keep right                                   |
| 8.51617e-06   | Speed limit (80km/h)                         |
| 7.92366e-06   | Roundabout mandatory                         |


IMAGE #4: Children crossing  
![alt text][image8]

| Probability   |     Prediction                       | 
|:-------------:|:------------------------------------:| 
| 0.843718      | Road work                            | 
| 0.156268      | Dangerous curve to the right         |
| 5.41687e-06   | Go straight or right                 |
| 3.17688e-06   | Speed limit (60km/h)                 |
| 2.21323e-06   | Keep right                           |


IMAGE #5: Right-of-way at the next intersection 
![alt text][image9]

| Probability   |     Prediction                       | 
|:-------------:|:------------------------------------:| 
| 1.0           | Right-of-way at the next intersection| 
| 4.13463e-38   | Beware of ice/snow                   |
| 0.0           | Speed limit (20km/h)                 |
| 0.0           | Speed limit (30km/h)                 |
| 0.0           | Speed limit (50km/h)                 |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Feature map shows the output of the new test image #1 (Speed limit 70 km/h). Convolution layer is converting the image to 1x4 feature map. As highlighted in the image, all 4 channels are learining different part of the areas to extract information from image. 

![alt text][image4]
