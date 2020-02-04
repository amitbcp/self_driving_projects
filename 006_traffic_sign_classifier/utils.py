import cv2
import numpy as np


class ImagePreprocessor:

  def __init__(self, equalize=True, normalize=True):

    self.equalize = equalize
    self.normalize = normalize

  def equalize_im(self, image):
    image = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))[0]
    image = cv2.equalizeHist(image)
    return image

  def normalize_im(self, image):

    mini, maxi = np.min(image), np.max(image)
    image = (image - mini) / (maxi - mini) * 2 - 1
    return image

  def preprocess_im(self, image):

    if self.equalize:
      image = self.equalize_im(image)
    if self.normalize:
      image = self.normalize_im(image)

    if self.equalize:
      return np.expand_dims(image, axis=2)
    else:
      return image

  def rotate(self, image, angle=15):
    angle = np.random.randint(-angle, angle)
    M = cv2.getRotationMatrix2D((16, 16), angle, 1)
    return cv2.warpAffine(src=image, M=M, dsize=(32, 32))

  def translate(self, image, pixels=2):
    tx = np.random.choice(range(-pixels, pixels))
    ty = np.random.choice(range(-pixels, pixels))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(src=image, M=M, dsize=(32, 32))

  def random_bright(self, image):
    eff = 0.5 + np.random.random()
    return image * eff

  def preprocess(self, data):
    dataset = np.array([self.preprocess_im(img) for img in data])
    return dataset

  def augment(self, images, count):
    augmented = []
    while True:
      for image in images:
        if len(augmented) == count:
          return augmented
        image = self.random_bright(image)
        image = self.rotate(image)
        image = self.translate(image)
        image = self.normalize_im(image)
        #print(image.shape)
        augmented.append(np.expand_dims(image, axis=2))
