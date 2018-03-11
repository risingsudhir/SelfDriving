## Vehicle Detection Project

This projects detects on road vehicles using a trained classifier and video streaming from camera, mounted on the centre of the vehicle.

* A Support Vector Machine classifier has been trained to classify images as car or non-car images. 
* Classifier has been trained om Histogram of Oriented Gradients (HOG) feature set, color histograms feature set, and binned color feature set.
* Region of interest from frame image is extracted by sliding window to locate vehicles in the frame. 
* Image heatmap is generated for detected positions and bounding boxes are drawn on the frame.

[//]: # (Image References)
[image1]: ./output_images/1.png
[image2]: ./output_images/2.jpg
[image3]: ./output_images/3.jpg
[image4]: ./output_images/4.jpg
[image5]: ./output_images/5.png
[image6]: ./output_images/6.png
[image7]: ./output_images/7.png
[image7]: ./output_images/8.png
[video1]: ./test_videos_output/project_video.mp4



### Vehicle Classifier

ImageFeatureExtractor class implements the feature extraction as an input for classifier training. Folowing set of features extracted from the image from RGB color space:

* Color bin spatial feature
* Color channels histogram
* Histogram of Oriented Gradients (HOG)

All three feature vectors are combined together. Images below are shown for two different classes (car and non-car images) of images and corresponding HOG feature vector.

![alt text][image7]

![alt text][image8]

Feature vector parameters were selected to keep the balance of feature vector and training complexity. RGB color channel has been used for feature extraction.

To train the classifier, a Support Vector Machine with linear kernal has been used. 'C' value optimization is done using grid search method. SVM with linear kernal and 'C' value of 10 has been selcted as best fit for the image data.

Training and Test set accuracy is 99.97% and 99.03%, respectively.


### Vehicle Detection

VehicleDetector class implements the detection of vehicle in an image frame. Vehicle detector performs following operations to find vehicle in the image frame: 

* Finds ragion of interest using sliding window
* Each window is then passed to classifier to identify if image is of a vehicle or non-vehicle 
* Detected vehicle windows are passed to image heatmap generator to filter out noise and smooth detection of vehicles.
* Image heatmap generator finds coordinates of the detected vehicles.

Region of interest and window sliding positions are determined by the fact that Camera is mounted on the centre of the vehicle.

Below images are shown with region of interest and detected vehicles:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]


### Video Implementation

Image pipeline has been extended to work with video frames. Each video frame is analysed and tracked for vehicles. For optimization, once a vehicle has been detected by heatmap, detector will skip next 20 frames and analyze frames for video in 21th frame. This is helping in faster processing of the video frames with minimal loss of accuracy. 

Output video is here [output_video](./test_videos_output/project_video.mp4)


### Discussion

There are atleast two possible optimizations for the current pipeline:  

* as suggested in the lecture videos, HOG feature for single frame can be computed separately instead of calculating feature vector for individual window image. This will speedup frames processing.

* RGB feature is a bit sensitive to shadows. In my last project for advanced lane finding, I was able to filter shadow effects from R channels by taking the votes from S (HLS) and Gray channels. A similar technique can be applied to make the pipeline more reobust to shadows.
