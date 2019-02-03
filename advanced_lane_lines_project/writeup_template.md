## Self-Driving Car Engineer Nanodegree


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Step 0 : Necessary Imports

All the necessary packages for the project are imported in the very start of the project. Also a helper function to `plt_images()` is included to abstract common graph plots.


### Step 1 : Camera Calibration

The code for this step is contained in the following funtion in the 3rd cell of the IPython notebook located in "./examples/example.ipynb"
* **calibrate_camera()** - This method goes over multiple raw images to detect ChessBoard corner points used for calibration

The  "object points", from the calibrate_camera() which will be the (x, y, z) coordinates of the chessboard corners in the world. Here,the assumption is tht the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time it successfully detects all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

### Step 2 : Undistort Images
The code for this step is contained in the following funtion in the 5th cell of the IPython notebook located in "./examples/example.ipynb"
* **undistort()** - This method remove distortion from images base on the calibrtion object points of the camera.

The output `objpoints` and `imgpoints` is used tp compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

### Step 3 : Transforms for Binary Threshold Images
Basic image transformation to detect/highlight the lane in the image/video were applied as mentioned below.

0. **`Mask Region of Interest in the image`** : This is used to avoid other cars/vehicle passing by from affecting our line detection
1. **`Sobel Filters for edge detection`** : We take use basic Filters over the image in both X & Y direction to detect edges.
2. **`Gradient Magnitude`** : We take the magnitude of the gradient inn X & Y direction and define a threshold for detectio
3. **`Directional Gradient`** : We also define the direction of the Gradient and limit them from 0 to np.pi/2 for the project
4. **`Color Thresholds in different color spaces`** : The image is transformed to the  LUV(for detecting `white` lines) & Lab(for detecting `yellow` lines) instead of standard HLS space which seems to be adding noise.
5. **`Combining Thresholds`**  : Finally all the transforms where combined to retrive a final images. The function used for this was `combine_threshs()`

The thresholds where defined manually after trying over multiple images and tuning them to get a better results.

![alt text][image3]

### Step 4 : Perspective Transform

Perpective transform is performed over an undistorted image. In the project `birds_eye()` is used for the transformation. It take a couple of other inputs to determine whether to perform distortion or not, finally displaying the output.

The source (`src_coordinates`) and destination (`dst_coordinates`) points were hardcoded in the project but provided good output/conversion of hte raw image. The warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

### Step 5 : Detect Lane Lines

At first the histogram was computed for the image, following which the left & right lanes where detected based on the spikes on the left/right section of the image.

Instead of computing blind search for every search, after the inital few frames, the lane search was limited only to the near by region of hte previous lane fits to fasten app the processing. Once again if no lane was detect, a blind search was used via the histogram mapping.
The following functions were used for the following  - 
1. **`get_histogram()`** - This functions computes the histogram mapping of any input image.
2. **`detect_lines()`** -  This function performs a blind search on the frame to detect the left/right lane
3. **`detect_similar_lines()`** - This function relies on the previously detected lanes to find lane points in the vicinity. If not found they perform a blind search again using `detect_lines()`

![alt text][image5]
### Step 6  :  Radius of Curvature
This has been implemented in the `curvature_radius()` function of the project. Here we do the following steps -
1. Reverse map the lane positions to match Top-to-Bottom Y points
2. Convert pixel space to world space in meters
3. Fit a polynomial in the World-Space
4. Calculate the Radius of Curvature using the formula.

### Step 7 : Calculating Vehicle Position

This has been implemented in the `car_offset()` function. It computes the car location with respect to the mid-point of the image frame and the location of the left-right lane.


### Step 8 : Warp the detected lane boundaries back onto the original imagen

This was implemented via the `draw_lane()` and `add_metrics()` function of the project.
The steps involved were -
1. Plot the polynomials on the warped image.
2. Fill the space between the Polynomials to show the lane.
3. Perform inverse Perspective Transform to get the original image from the warped image.
4. Adding the calculated metrics on the image.

![alt text][image6]

---

### Pipeline (video)

After completing each of the above steps, they need to be orhestrated properly to make a singular Pipeline for processing images and videos alike.
For this the class **`ProcessImage`** was created to handle the flow at one place. This also encourages addition & removal of processing steps easily from the pipeline. This pipeline has thus processes videos frame-by-frame, to simulate a process of real-time image stream from a actual vehicle.

The `ProcessImage` pipeline first evaluates whether or not a lane was detected in the previous frame. 
* If not, it performs a blind search over the image to find the lane. 
* If the lane was detected in the previous fram, it only only searches for the lane, in close proximity of the previous lane (polynomial of the previous frame). 
* This enables the pipeline/system to avoid scanning the entire image and build high-confidence (enabling more fault tolerant) as new location is based on the previous location.
* When the pipeline fails to detect a lane pixel based on the previous detected pixels, it reverts back to blind search mode to scan the entire image for non-zero pixel via the image histogram.
* This also boosts the time performance of or pipeline.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
#### Challenges Faced -

1. With the challenge and harder challenge video, it pipeline performace was detected. This showed that the preprocessing values to do not hold true for under all circumstances. 
2. The test images used for fine-tuning the thresholds had fairly good images under proper lighting. Images uder bad lighting needs to be included to tune the parameters further.
3. Perspective Transform points where chosen manually and hence performed poorly in the harder challenge videos.

#### Futhure Work
1. At first, more test images of different conditions needs to be included.
2. Other pre-processing transforms like Gaussian Blur, Dilation & Erosion needs to be included in the pipeline
3. The Similar Detection of lanes can be further improved to keep a memory of more the ust one previosu lane detection.
4. A running average for the lane detection can be used to make smooth polynomials.
5. A clipping can be used when the Radius of Curvature / Vehicle offset makes a sharp change compared to it's previous values as that's a wronf detection in general evident from the harder challenge videos.
