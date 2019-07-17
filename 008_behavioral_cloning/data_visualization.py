import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from training import TrainingPipeline
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm


# Check and show the train, validation data steering feature
def show_steering(y_train, y_valid, image_name):
  '''take train and validation data label-steering and visualize a histogram.
    input: y_train : train set label,
           y_valid : validation set label,
    output: Histogram of labels'''

  max_degree = 25
  degree_per_steering = 10
  n_classes = max_degree * degree_per_steering
  fig, axes = plt.subplots(2, 1, figsize=(8, 8))
  plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
  ax0, ax1 = axes.flatten()

  ax0.hist(
    y_train,
    bins=n_classes,
    histtype='bar',
    color='blue',
    rwidth=0.6,
    label='train')
  ax0.set_title('Number of training')
  ax0.set_xlabel('Steering Angle')
  ax0.set_ylabel('Total Image')

  ax1.hist(
    y_valid,
    bins=n_classes,
    histtype='bar',
    color='red',
    rwidth=0.6,
    label='valid')
  ax1.set_title('Number of validation')
  ax1.set_xlabel('Steering Angle')
  ax1.set_ylabel('Total Image')

  fig.tight_layout()
  plt.savefig(image_name)
  plt.show()


def load_data(data):
  images, steering_angles = list(), list()
  for data in tqdm(data):
    image, steering_angle = training.process_batch(data, append=True)
    images.extend(image)
    steering_angles.extend(steering_angle)

  X, y = np.array(images), np.array(steering_angles)

  return X, y


if __name__ == "__main__":
  file = './data/driving_log.csv'

  training = TrainingPipeline(base_path='./data', epochs=1)
  training.import_data()
  training.split_data()

  #split not the data just their names
  # We don't need to test images because it is a regression problem not classification.
  #   train_samples, validation_samples = train_test_split(
  #     training.data, shuffle=True, test_size=0.2)
  #print(training.training_data.shape)
  X_train, y_train = load_data(training.training_data)
  #   images, steering_angles = list(), list()
  #   for data in training.validation_data:
  #     image, steering_angle = training.process_batch(data, append=False)
  #     images.extend(image)
  #     steering_angles.extend(steering_angle)

  X_valid, y_valid = load_data(training.validation_data)

  #show
  show_steering(
    y_train, y_train, image_name='augmented_dataset_distribution.png')
