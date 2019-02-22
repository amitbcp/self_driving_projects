# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/Train_Dataset_Distribution.png "Training Dataset Distribution"
[image2]: ./output_images/Test_Dataset_Distribution.png "Test Dataset Distribution"
[image3]: ./output_images/Validation_Dataset_Distribution.png "Validation Dataset Distribution"
[image4]: ./output_images/visualize_training_data.png "Visualize Training Images"
[image5]: ./output_images/preprocessed_images.png "Preprocessed Images"
[image6]: ./output_images/training_accuracy.png "Training Loss Graph"
[image7]: ./output_images/training_loss.png "Training Loss Graph"
[image8]: ./output_images/confusion_matrix.png "Confusion Matrix"

[image9]: ./input_images/image_1.jpg ""
[image10]: ./input_images/image_2.jpg ""
[image11]: ./input_images/image_3.jpg ""
[image12]: ./input_images/image_4.jpg ""
[image13]: ./input_images/image_5.jpg ""

[image14]: ./output_images/new_prediction.png "Web Images Prediction"
[image15]: ./output_images/spped_limit_80_visualisation.png "Image"
[image16]: ./output_images/spped_limit_80_feature_map2.png "Convolution 1"
[image17]: ./output_images/spped_limit_80_feature_map1.png "Convolution 2"


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Exploratory visualization of the dataset.

We can visualize the class distribution of the dataset. Hence we can infer that all the 43 classes are not equally balanced and there is a lot of skewness in the data. Though it is also to be noted that the distribution is same across the train/valid/test data case.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing , I used the following techniques - 
1. Equalized `Y` channel in the YUV color space : This works as `Y` denotes the luminance where as the U & V are the color. Since the Traffic signs denote meaning from the shape rather than color,it makes sense that a converting to Grayscale which takes an average of the Color Channel.
2. Normalizing the image using the Min & Max Normalization .
3. Finally added the channel dimension for the Neural Network.

Data Augmentation was also used to increase the training data majorly because of the following : 
1. The classification is for 43 classes, and it is important to have sufficient representation in each class.
2. We don't want the classifier to be biased for any particular class due to training data shortage.
3. Since Traffic Signs classification is a real world problem, the model should be trained to handle image from any angle and lighting conditions.

![alt text][image5]

The following techniques were used for the data augmentation :
1. **`Random Rotation`** : Randomly rotating the image in a fixed range.
2. **`Wrap`** : Created a Affine Wrap for the image
3. **`Brigtness`** : Changing the base brightness of the image.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Image   							| 
| Convolution 1 : 3x3     	| 1x1 stride, VALID padding, outputs 30x30x12 	|
| RELU					|												|
| Convolution 2 : 3x3     	| 1x1 stride, VALID padding, outputs 28x28x24 	|
| Max pooling	 1     	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 3 : 5x5	    | 1x1 stride, VALID padding, outputs 10x10x36     									|
| RELU					|	
| Convolution 4 : 5x5	    | 1x1 stride, VALID padding, outputs 6x6x48     									|
| RELU					|												|
| Max pooling	  2    	| 2x2 stride,  outputs 3x3x48  				|
| Flattening		|   The output of Max Pool 1 (i.e. from 2nd Convolution) & Max Pool 2 (i.e. from 4th Convolution) are concatenated. This is skip connection/residual connections      									|
| Fully connected	1	| 5136 x 512       									|
| RELU					|												|
| Dropout					|												|
| Fully connected	2	| 512 x 256       									|
| RELU					|												|
| Dropout					|												|
| Fully connected	3	| 256    x 43   									|
| Softmax				|         									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

With the above explained model architecture, along with the augumented data , I used to following for training : 
1. AdamOptimizer - This enables faster convergence compared to gradient descend, along with it's inherited RMSprop & Momentum application
2. Learning Rate - A very low learning rate of **0.0005** was used with auto-decay to reduce it further as the training progesses.
3. Batch Size - Batch Size of 64 was used as a multiple f 32 to enable more stable learning and average updated during back-propagation.
4. Epoch - 100 Epochs where chosed for training but it can be seen that the model learned in less than 10 Epochs.
5. Dropout - Keep Probability of **0.7** was used instead of standard 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.9%
* test set accuracy of 97.04%

![alt text][image6]
![alt text][image7]

If an iterative approach was chosen:
* A Deep CNN was chosen as the first architecture with 2 Convolution Layers followed by Fully connected layers
* It reached a accuracy of 68 %
* The model had low accuracy on both sets indicating under fitting. Which was evident from the shallowness of the Network.

Finally I explored residual networks.
* A simple residual network was implemented which created a skip connection to the final convolution layer for flattening.
* It seems relevant for Traffic Signs as , the signs have very few sections that are colured. They have very distinct and clear boundaries and consistent color. This leads to the dead neurons because of no gradients flowing through them during back-propagation. Hence the adding skip-connections is a legit option.
* The result can be seen from the training, that the model achieves high accuracy in just a couple of Epochs.
 
![alt text][image8]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web along witht there probabilty distribution for prediction

![alt text][image9]![alt text][image10]![alt text][image11]
![alt text][image12]![alt text][image13]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the feature map visualisation we can infer the the model learned the followings features based on which it identifies/classifies the images -
1. Boundaries and edges - We can clearly see that differnt features of the convolutions are getting acctivated by different section of hte image boundary.
2. Blobs of activations - We can also see from the second convolution that the blobs of pixel are activating the Filter map which shows the network learning basic feature detection , which it uses to detect lines/edges/circles.

