import cv2
import numpy as np


def bgr2rgb(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flip_img(image):
  return cv2.flip(image, 1)


def crop_img(image):
  cropped = image[60:130, :]
  return cropped


def resize(image, shape=(160, 70)):
  return cv2.resize(image, shape)


def crop_resize(image, shape=None):
  cropped = crop_img(image)
  if shape is None:
    resized = resize(cropped)
  else:
    resized = resize(cropped, shape)
  return resized
