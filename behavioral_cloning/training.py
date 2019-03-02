import csv
import cv2
import utils
import argparse
import numpy as np
from model import network
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TrainingPipeline:
  def __init__(self, model=None, base_path='', epochs=2):
    self.data = []
    self.model = model
    self.epochs = epochs
    self.training_data = []
    self.validation_data = []
    self.correction_factor = 0.2
    self.base_path = base_path
    self.image_path = self.base_path + '/IMG/'
    self.driving_log_path = self.base_path + '/driving_log.csv'

  def import_data(self):
    print("Importing data ...")
    with open(self.driving_log_path) as csv_data:
      reader = csv.reader(csv_data)
      # Skipping the column names in the CSV file
      next(reader)
      for row in reader:
        self.data.append(row)

    return self.data

  def process_batch(self, batch, append=True):
    steering_angle = np.float32(batch[3])
    images, steering_angles = list(), list()

    for image_path in range(3):
      image_name = batch[image_path].split('/')[-1]
      image = cv2.imread(self.image_path + image_name)
      image_rgb = utils.bgr2rgb(image)
      #image_resized = utils.crop_resize(image_rgb)

      images.append(image_rgb)
      if image_path == 1:
        steering_angles.append(steering_angle + self.correction_factor)
      elif image_path == 2:
        steering_angles.append(steering_angle - self.correction_factor)
      elif image_path == 0:
        steering_angles.append(steering_angle)

      if append:
        #Appending data
        image_flipped = utils.flip_img(image_rgb)
        images.append(image_flipped)
        steering_angles.append((-1) * steering_angle)
    return images, steering_angles

  def data_generator(self, data, batch_size=32):
    num_samples = len(data)

    while True:
      shuffle(data)

      for offset in range(0, num_samples, batch_size):
        batches = data[offset:offset + batch_size]
        images, steering_angles = list(), list()

        for batch in batches:
          images_augmented, angles_augmented = self.process_batch(batch)
          images.extend(images_augmented)
          steering_angles.extend(angles_augmented)

        X_train, y_train = np.array(images), np.array(steering_angles)
        yield shuffle(X_train, y_train)

  def split_data(self, ):
    train, validation = train_test_split(self.data, test_size=0.2)
    self.training_data, self.validation_data = train, validation

  def training_data_generator(self, batch_size=32):
    return self.data_generator(data=self.training_data, batch_size=batch_size)

  def validation_data_generator(self, batch_size=32):
    return self.data_generator(
      data=self.validation_data, batch_size=batch_size)

  def run(self):
    self.split_data()
    history_object = self.model.fit_generator(
      generator=self.training_data_generator(),
      validation_data=self.validation_data_generator(),
      epochs=self.epochs,
      steps_per_epoch=len(self.training_data) * 2,
      validation_steps=len(self.validation_data),
      verbose=1)

    self.model.save('model_1_ep_' + self.epochs + '.h5')
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('Training_Loss_md_1_ep_' + self.epochs + '.png')
    #plt.show()


def main():
  training = TrainingPipeline(model=network(), base_path='./data', epochs=1)
  training.import_data()

  training.run()


if __name__ == "__main__":
  main()
