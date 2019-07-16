**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images/solidWhiteCurve.jpg "Original Image"
[image3]: ./test_images_output/solidWhiteCurve.jpg "Lane Detected Image"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The Lane Line detector pipeline was implemented based on the following : 
#### Assumptions
  1. Lanes on the road are straight lines.
  2. The image/video input are of the dimension of (960,540). If not they are resized to the same
 
#### Implementation
  1. Declare the Region of Interest as a Global Variable.
  2. Determine it's central point, to be used later.
  3. Resize the image to (960,540) dimensions
  4. Convert image to a grayscale image.
  5. Apply Gaussian Blur to smoothen the image and reduce noise.
  6. Apply Canny Edge Detection on the smoothened image.
  7. Select the Region of Interest with the pre-defined co-ordinated of Step 1, to focus only the image having the lane.
  8. Apply Hough Transform, on the masked image to join the lane edges to lines.
  9. Change the draw_lines() function as follows to extrapolate the lanes -
      1. Calculate the Slope & Intercept of every line detected by Hough Transform.
      2. Add co-ordinated to Left/Right lane list based on the position of the co-ordinate from the center of the "Region of Interest"
      3. Calculate the Max & Min slope from the list of lines detected from Hough Transform.
      4. Extrapolate the left_lane/right_lane co-ordinates using the **np.polyfit()** to determine the overall slope & intercept.
      5. Get the line co-ordinated with these slope & intercept, along with the min/max Y-co-ordinate values of the Region of interest.
  10. Plot the lines over the image to get the Lane Lines
    
  
![alt text][image2]
![alt text][image3]


### 2. Identify potential shortcomings with your current pipeline

1. Lane Lines sometimes flicker along the top edge as there is no smoothing/ threshold to change or hold memory of previous detection.
2. This model does not work for curved roads.
3. The model draws lane for the Region of interest with the same confidence.

### 3. Suggest possible improvements to your pipeline

1. **Smoothing** - The Lane Lines can be made to have memory, for which any abrupt change in predicted lane slope can be stopped, hence helping flickering.
2. **Slope** - Slope of the Hough Lines can be used to distuingush Left/Right Lane rather than the central point in the Region of Interest.
3. Smaller Lane predicted lines can be used & merged for curved roads as given in the challenge.
