[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

This project, we'll combine our knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project is be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/amitbcp/deep_learning_projects.git
cd deep_learning_projects/001_facial_keypoint
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__:
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__:
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```

	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.

	- __Linux__ or __Mac__:
	```
	conda install pytorch torchvision -c pytorch
	```
	- __Windows__:
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the P1_Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.


## Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd 001_facial_keypoint
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-nd` environment by clicking `Kernel > Change Kernel > cv-nd`.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality and answer all of the questions included in the notebook. __Unless requested, it's suggested that you do not modify code that has already been included.__


## Evaluation

Your project will be reviewed against the project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Intuition

### 1: What optimization and loss functions did you choose and why?

**Answer**: I preferred MSE loss has we are predicting the key-points over here which is a regression problem when we evaluate with the real key-points. The distance(/difference) between the real and predicted serves as a good measure of the error made by the Model.

**Optimization** - I preferred Adam optimizer for it's it's computational effectiveness and better convergence compared to SGD. Since it encapsulates RMSProp & AdaGrad, it brings the effectiveness along with default parameters which suggested good results.


#### 2: What kind of network architecture did you start with and how did it change as you tried different architectures? Did you decide to add more convolutional layers or any layers to avoid overfitting the data?

**Answer**: I started with a simple architecture with 2 Convultion , 1 Dropout & 1 FC layer. Though I reached pretty low loss, the model seemed to overfit as it the test key points were almost always constantly shifted from the original points.

Then I decided to make the network deeper by adding CNN , Batch Normalization on evrey layer, dropout on every layer and a couple of FC at the end to avoid overfitting & proper scaling of weights.


### 3: How did you decide on the number of epochs and batch_size to train your model?

**Answer**: I decided batch_size based on speed of training with almost same output loss. I decided the epochs on the same basis along with visualizing loss not falling after a certian epoch.

Also, batch size affected the wiggles in the training loss,i.e. low batch sizes made the loss jump around where as larger batch sizes ensured a smoother loss curve.

### 4: Choose one filter from your trained CNN and apply it to a test image; what purpose do you think it plays? What kind of feature do you think it detects?

**Answer**: The above Kernel feature map does not provide any particular shape or feature that it gets activated too. But once we look at the images after the kernel, we can observe that the kernel is highligting humna skin darkly compared to the clothes or surrounding on all the three test images.

Now, we can relate that the filter is generic enough and acting on intensity change and edge detection horizontally.
