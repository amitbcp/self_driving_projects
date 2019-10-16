# **Behavioral Cloning Project**
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/architecture.png "Model Visualization"
[image2]: ./output_images/original_dataset_distribution.png "Data Distribution"
[image3]: ./output_images/augmented_dataset_distribution.png "Further Augmented Data Distribution"
[image4]: ./output_images/augmented_dataset_distribution_v1.png "Augmented Data Distribution"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture was inspired from NVIDIA to a certain extent but later with the help of the results, it could be seen that even a slightly simpler model works good.

The model is defined in the ```model.py``` script. It could be seen that the script has two function with the only difference in terms of addition of Dropout to make reduce overfitting.
The architecture has the following characteristics. -
  1. The Kernels used were 5x5 & 3x3. The depth size keeps on increasing from 24-36-48-64 till the end in the given order.
  2. RELU has been used as the Neural Network Activation  function to add non-linearity.
  3. Dropout is included in the last dense layers to avoid overfitting of the model.
  4. It should also be noted that no Max-Pool or Avg-Pool layers were used. Instead strides of 2 was used to reduce the dimensionality & parameters in the network.
  5. For loss , Mean Squared Error was used and, ADAM's optimizer for backprogation and convergence of the model.

#### 2. Attempts to reduce overfitting in the model

It was observed that the model started overfitting on the dataset highly on increasing the number of epoch's, i.e. training loss kept on decreasing while validation loss remained same.
The methods used for the same were -
  1. The entire dataset was divided into two separate-parts. Training/Validation which were kept separate. The division was 80%-20% for Training-Validation.
  2.Tracking Loss across the training over both training data-set & validation-set
  3. Dropout was used in the fully connected layers of the model.
Finally the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned automatically during the training.

#### 4. Appropriate training data

  1. Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
  2. Apart from using only the center camera image, the images from the left/right camera was used for trainig. The steering angle was accordingly adjusted for the same , so the the car recovers if it moves to much in a particular direction going off the road.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it small yet powerful.
Initially I thought of using some state-of-the art architectures but soon realised that those would be an overkill given this is an regression problem and not a classfication problem.

The model was designed based on the following nature of the problem and properties -
  1. Since it was a regression problem, choosing the error metric was simple to ```MSE``` and `ADAMs` optimizer.
  2. Since the camera captures the image from the car is the front view, it made sense using filters of size 5x5 in the begining rather towards the end of the network.
  3. Larger kernel size increases the optical view of the convolution helping it recognize the contnious filter on a larger scale than a 3x3
  4. Since the road is expected to have less frequency distribution, a 5x5 works better followed by 3x3 to bottleneck the convolution.
  5. This also helps in reducing the parameters in the network.
  6. Based on the above intuition that the road has less frequency distribution, instead of using Max-Pool layers, the convolution was done with a stride of 2 in the first few layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I first augmented the training data further to have a better distribtuion. It can be visualised below -

##### Orginial Data Distribution

![alt text][image2]
##### Augmented Data Distribution

![alt text][image4]
##### On further Augmenting Data

![alt text][image3]


The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For the training data, I used the default data provided by Udacity as using it along with Data Augmentation techniques, I was able to train the network pretty well. To augment the data I used the following -
  1. Included the Left & Right camera images to help the car recover from the side and adjusting the steering angle
  2. Flipped the images & it's steering angle. That is similar to driving on the track from end to start.

After the collection process, I had 12k number of data points for training. I then preprocessed as follows -
  1. The image was converted to RGB Color Space
  2. The image was cropped 60 pixels from the top to remove the scenery as that acts as unnecessary information/noise for the network
  3. The image was cropped 30 pixels from the bottom to remove the car front hood and eliminate noise.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the model did not show much improvement over increasing the epoch, rather it started overfitting and crashing the car when the epoch was made to 4 in one of the experiments.ADAM optimizer so that manually training the learning rate wasn't .necessary.
