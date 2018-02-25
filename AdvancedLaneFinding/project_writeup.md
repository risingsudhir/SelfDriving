## Advanced Lane Finding Project

This project finds lane lines onto the road and measures their curvature and vehicle postion on the road.

* Camera calibration matrix and distortion coefficients are computed by given a set of chessboard images.
* Distortion correction is applied on raw images.
* Color transforms, gradients are used to create a thresholded binary image.
* Perspective transformation is used to rectify binary image ("birds-eye view").
* Lane pixels are detected and fit to find the lane boundaries.
* Curvature of the lane and vehicle position are calculated.
* Detected lanes are warped onto the original frame image.
* Lane boundaries and estimations are shown on the frame image.

[//]: # (Image References)

[image1]: ./output_images/1.png "Distortion Correction"
[image2]: ./output_images/2.png "Binary Transformation"
[image3]: ./output_images/3.png "Perspective Transformation"
[image4]: ./output_images/4.png "Lane Pixel Histogram"
[image5]: ./output_images/5.png "Overlapped Sliding Window"
[image6]: ./output_images/6.png "Lane Projection"
[image7]: ./output_images/7.png "Directed Search"
[image8]: ./output_images/8.png "Directed Search Lane Projection"
[video1]: ./test_videos_output/project_video.mp4 "Video"


### Camera Calibration

'CameraCalibration' class is responsible for camera calibration and distortion correction. To calibrate the camera, 9x6 black and white chess board images have been used. Calibration images are stored in 'camera_cal' folder.

To calibrate the camera, object points are marked as actual chess board corners and image points are detected chess board corners on the image. All image points are collected for calibration. During calibration, disortion matrix and coefficients are calculated which are used to correct distortion.


### Image Transformation

'ImageTransformation' class is responsible for transforming the image for lane detection. Following steps are performed during image transformation - 

#### 1. Distortion Correction

Frame image is undistorted using calibrated camera. Below image shows the camera distortion correction: 

![alt text][image1]

#### 2. Binary Tranformation

Three different types of binary images are constructed using different colour channels and gradient thresholds. First binary image is constructed using Gray(G) color channel with gradient threshold. Second binary image is constructed using Red(R) color channel with threshold. Third binary image is constructed using Saturation(S) color channel from HLS channel with threshold.

To construct the final binary image, following algorithm is used: 
 - pixels appearing in both S-Channel and G-Channels are always included.
 - pixels in R Channel needs to be supported by pixes in S-Channel or G-Channel.

This algorithm is helping in filtering out noise when R channel is weak or noisy, and providing a robus binary transformation.

Here is an example of binary transformation: 

![alt text][image2]


#### 3. Perspective Transformation

Perspective transformatin is used to change the view to top down presentation. Following source and destination points have been used for perspective transformation:

- Bottom Lane points to Top Lane Points [Lane-1-Start, Lane-2-Start, Lane-1-End, Lane-2-End]
        source:  (200, 700), (1100, 700),(590, 450), (690, 450)
    destination: (300, 720), (1000, 720), (300, 70),  (1000, 70)

Image below shows the perspective transformation of the lane lines.

![alt text][image3]

#### 4. Lane Lines Detection

Initially, histogram of pixels is used to identify lane pixels. Peaks from left and right half of the histogram are used to locate left and right lane points. 

![alt text][image4]

These starting points are used by sliding window search to identify lane lines vertically. In this project, I have modified the sliding window search to use 'overlapped sliding window search'. This version of sliding window search takes help from the previous window and controls the deviation across windows.

Also, since both lane lines are parallel in nature, deviation in window mean position change is also controlled by measuring magnitude of change in both windows' positions. This helps in giving direction to each other if one line has weak support of pixels.

Image below shows overlapped sliding window search of lane lines.

![alt text][image5]

Once lane lines are identified through overlapped window search, next frames are search using 'directed' search method. Directed search performs seach of lane lines within margin of lane lines detected in previous frames. 

Image below shows how directed search has defined the region of interest based on initial overlapped sliding window search.

![alt text][image7]


#### 5. Radius of Curvature and Vehicle Position

Image pixels are converted into meter to calculate radius of curvature and vehicle position w.r.t. centre of the lane lines.  LaneSearch.findRadius and LaneSearch.findVehiclePosition methods is used to calculate radius and vehicle positions.


#### 6. Projecting Lane Lines on Current Frame

Second order polynomial is used to fit the lane lines. Fitted lane pixels are then drawn and inverse perspective transformation is performed to project these lines onto original frame.

Image below shows the projection of lane lines onto frame image.

![alt text][image6]


### Video Streaming

To detect lane lines in a video, each frame is processed individuallly. Output of provided video is here: 

![alt text][video1]


### Discussion

I think binary image transformation techniques and overlapped window search are providing a robust result. I have not implemented smoothing and averaging out lane positions across frames, which can further improve results.
