# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg
[image2]: ./test_images_output/solidWhiteRight.jpg
[image3]: ./test_images_output/solidYellowCurve.jpg
[image4]: ./test_images_output/solidYellowCurve2.jpg
[image5]: ./test_images_output/solidYellowLeft.jpg
[image6]: ./test_images_output/whiteCarLaneSwitch.jpg
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipleline to identify lane lines in an images consisted of following operations: 
* gray scale transformation of the coloured and gaussian smoothing to reduce noise in intensity transition
* edge detection using Canny Edge detection algorithm
* identify the area of interest in which lanes needs to be detected by applying mask on detected edges.
* detect lines using Hough Line detection alorithm
* overlay lines on the original image.

For video streaming, single left and right lanes were shown by modifying draw_lines function. This function was modified to separate left and right lanes based on gradient of each line. Further, lines with similar gradients were combined together to draw a single line.

Below are results of the pipeline.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


### 2. Identify potential shortcomings with your current pipeline


standard pipeline does not work well for narrow vision. There is an assumption of wide range vision, with equidistance focus from left and right lanes. This is evident in the challenge video where standard pipeline failed to detect lanes without much of noise.

Another shortcoming could be the way draw_line method makes a assumption if gradient differences between left and right lanes. I thik it will fail to separate two perfectly parallel lef and right lanes.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to to adjust the area of interest based on the direction of the vehicle or the sensor. At the moment, it is static and does not suits only a specific range of images.

Another potential improvement could be to apply interpolation based on the location and perpendicular distance between two lane lines.
