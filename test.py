# imports
from functions.pytorch_models import hed_cnn, Trainer, pretrained_weights, hed_predict
from functions.data_preprocessing import load_images, augment_images_kp, mask_to_uv
from functions.data_preprocessing import load_train_test_imagedata, save_train_test_imagedata
from functions.data_visualisation import plot_predictions, plot_refined_predictions, write_output_gif

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

import os, glob

import torch

from sklearn.model_selection import train_test_split

import imgaug as ia
import imgaug.augmenters as iaa

from ipywidgets import interact, fixed, IntSlider, FloatSlider, interact_manual

# Specify settings
imSize = (320,480) # specify the target image size (height, width)
imDir = './test_sites/*.jpg' # specify the directory with image data

imgPaths = [_ for _ in glob.glob(imDir)]

imData, _ = load_images(imgPaths,imSize)