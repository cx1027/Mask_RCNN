#!/usr/bin/env python
# coding: utf-8

# This is a simpified version of the other test notebooks that only does training.

# In[5]:


# Enable this for some more verbose info.
debugging = False
import tensorflow as tf
from keras import backend as K

K.clear_session()

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ## Common imports

# In[6]:


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
import collections
import logging

logging.basicConfig(filename='./log/log_maskRCnn_%s.log' % (datetime.datetime.today())
                    , format='%(asctime)s [%(levelname)-5.5s] %(message)s'
                    , level=logging.DEBUG)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
ROOT_IMAGE_DIR = os.path.abspath("images/")

# Path to the dataset (note this is a shared images directory)
# dataset_path = os.path.join(ROOT_IMAGE_DIR, "level1")
dataset_path = os.path.join(ROOT_IMAGE_DIR, "20200119")

models_dir = os.path.join(ROOT_DIR, "csci_e89_project/models/")
logging.debug("base dataset dir:%s" % dataset_path)
logging.debug("base modesl dir:%s" % models_dir)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_obj_0015.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# In[7]:


# for visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# For Config and Dataset 
sys.path.append(os.path.join(ROOT_DIR, "csci_e89_project/"))  # To find local version
import csci_e89_project.det as det

# ## Augmentation

# In[9]:


# The imgaug library is pretty flexible and make different types of augmentation possible. 
# The deterministic setting is used because any spatial changes to the image must also be 
# done to the mask. There are also some augmentors that are unsafe to apply. From the mrcnn
# library: 
# Augmentors that are safe to apply to masks: 
# ["Sequential", "SomeOf", "OneOf", "Sometimes","Fliplr", 
# "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]
# Affine, has settings that are unsafe, so always
# test your augmentation on masks

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

# http://imgaug.readthedocs.io/en/latest/source/augmenters.html#sequential
seq_of_aug = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)),  # random crops

    # horizontally flip 50% of the images
    # iaa.Fliplr(0.5), # Does not make sense for signs

    # Gaussian blur to 50% of the images
    # with random sigma between 0 and 0.5.
    iaa.Sometimes(0.4,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image.
    # Scale/zoom them from 90% 5o 110%
    # Translate/move them, rotate them
    # Shear them slightly -2 to 2 degrees.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-5, 5),
        shear=(-2, 2)
    )
], random_order=True)  # apply augmenters in random order

# # Train

# The default implementation uses the ResNet101 and FPN networks. These are very large and they recommend a GPU with 12GB or an Amazon P2 instance. Due to the size, I am training with a BATCH_SIZE of 2.

# In[10]:


logging.debug("dataset dir:%s" % dataset_path)
logging.debug("modesl dir:%s" % models_dir)

# In[11]:


# Setup configuration

# config = det.DetConfig('sign', ['sign', 'yield_sign', 'stop_sign', 'oneway_sign', 'donotenter_sign', 'wrongway_sign'])

config = det.DetConfig('obj',
                       ['obj', 'player_obj', 'monster_obj', 'surface_obj', 'box_obj', 'coin_obj'])
config.display()

# In[12]:


# Create the model
model = modellib.MaskRCNN(mode='training',
                          config=config,
                          model_dir=models_dir)

model.keras_model.summary()

# In[13]:


# Load the weights

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(models_dir, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
else:
    logging.debug("using existing: %s"% COCO_MODEL_PATH)

# For the coco dataset exclude the last layers because 
# it requires a matching number of classes
logging.debug("loading...")
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

logging.debug("loaded.")


# In[ ]:


def train(model, dataset_path, epochs=30):
    """Train the model."""

    # Create the train and val dataset.
    dataset_train, dataset_val = det.create_datasets(dataset_path + '/train', config)

    # Prepare them
    dataset_train.prepare()
    dataset_val.prepare()

    # Experiment with training options.
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    logging.debug("Training network heads")
    history = model.train(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=epochs,
                          layers='heads',
                          augmentation=seq_of_aug
                          )

    return history


# In[1]:

# try:
history = train(model, dataset_path, 30)
# except:
#     logging.debug("Unexpected error:{0}:".format(sys.exc_info()[0]))


# In[ ]:


logging.debug(history.history.keys())

# In[ ]:


import math


def plot_history(history):
    fig = plt.figure(figsize=(16, 10))

    n_history_loss = len(history.history)
    n_epochs = len(history.epoch)
    epoch = history.epoch

    # The loss is in pairs, one for train, one for val
    loss_stats = [k for k in history.history.keys() if 'val_' not in k]

    n_cols = 4
    n_rows = math.ceil(len(loss_stats) / n_cols)

    for i, k in enumerate(loss_stats):
        val_k = 'val_' + k
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.plot(epoch, history.history[k], label=k)
        ax.plot(epoch, history.history[val_k], label=val_k)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(str(i) + ' - ' + k)
        plt.legend(shadow=True, fancybox=True)

    fig.tight_layout()
    plt.show()


# In[14]:


# Plot training stats for each of the networks.
plot_history(history)

# # Inference

# In[15]:


# Find last trained weights
weights_path = model.find_last()
weights_path


# In[16]:


class InferenceConfig(det.DetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# inf_config = InferenceConfig('sign', ['sign', 'yield_sign', 'stop_sign', 'oneway_sign', 'donotenter_sign', 'wrongway_sign'])
inf_config = det.DetConfig('obj',
                           ['obj', 'player_obj', 'monster1_obj', 'surface_obj', 'box_obj', 'coin_obj'])

inf_model = modellib.MaskRCNN(mode="inference",
                              config=inf_config,
                              model_dir=models_dir)

inf_config.display()

# enabled for testing
# weights_path = os.path.join(models_dir, "sign20180508T1523_mask_rcnn_sign_0075.h5")

logging.debug("Using weights: %s" % weights_path)

inf_model.load_weights(weights_path, by_name=True)

# In[17]:


import glob


def detect_instance(class_names, image_test_dir):
    """
    class_names: list of class names of the dataset
    image_filenames: list of images to analyze
    """

    det_filenames = sorted(glob.glob(image_test_dir + '/*'))

    for f in det_filenames:
        logging.debug("Processing image {}".format(f))

        test_img = plt.imread(f)
        logging.debug(test_img.shape)

        plt.imshow(test_img)
        # visualize.display_images([test_img])

        # Included in the results from detect are the found:
        # class_ids,their scores and masks.
        results = inf_model.detect([test_img], verbose=1)[0]

        logging.debug("Objects detected: %i" % len(results['class_ids']))

        # Visualize results
        visualize.display_instances(test_img,
                                    results['rois'],
                                    results['masks'],
                                    results['class_ids'],
                                    class_names,
                                    results['scores'])

        logging.debug(results['class_ids'])


# In[18]:


import time


def detect_instances(class_names, image_test_dir):
    """
    class_names: list of class names of the dataset
    image_filenames: list of images to analyze
    """

    det_filenames = sorted(glob.glob(image_test_dir + '/*'))

    fig = plt.figure(figsize=(16, 10))

    n_cols = 3
    n_rows = math.ceil(len(det_filenames) / n_cols)

    for i, f in enumerate(det_filenames):
        logging.debug("Processing image {}".format(f))

        ax = plt.subplot(n_rows, n_cols, i + 1)

        test_img = plt.imread(f)

        start_time = time.time()

        # Included in the results from detect are the found:
        # class_ids,their scores and masks.
        results = inf_model.detect([test_img], verbose=1)[0]

        elapsed_time = time.time() - start_time

        logging.debug("Elapsed time: {:.4f}, Objects detected: {}".format(elapsed_time, len(results['class_ids'])))

        # Visualize results
        visualize.display_instances(test_img,
                                    results['rois'],
                                    results['masks'],
                                    results['class_ids'],
                                    class_names,
                                    results['scores'],
                                    ax=ax)

        logging.debug("class_ids:%s" %results['class_ids'])


# In[19]:


image_test_dir = os.path.join(dataset_path, "test/")

if debugging:
    r = detect_instance(inf_config.ALL_CLASS_NAMES, image_test_dir)
else:
    r = detect_instances(inf_config.ALL_CLASS_NAMES, image_test_dir)

# In[ ]:


# In[ ]:




