#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json
import os
import numpy as np
import pandas as pd
import glob
from numpy import genfromtxt
from keras import optimizers
#from keras.models import Model
import matplotlib.pyplot as plt
#import inputcsv as a
#import testshapes as a
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import keras as keras
from math import sqrt
from sklearn.utils import shuffle

from sklearn.manifold import TSNE
from sklearn import manifold

from keras import backend as K
from sklearn.metrics import mean_squared_error,median_absolute_error#mean_squared_log_error
from keras.models import load_model
#import cv2
import timeit
import math
from sklearn.metrics import jaccard_score

#from skimage.measure import structural_similarity as ssim
#import skimage.filters as filter
#from skimage.filters import threshold_local
class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))
    
def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

    
def calculateDistance(image1, image2):
    distance = 0
    image1 =   np.reshape(image1 ,(128, 128) )
    image2 =   np.reshape(image2 ,(128, 128) )
    i,j = np.unravel_index(image1.argmax(), image1.shape)
    f,h= np.unravel_index(image2.argmax(), image2.shape)
    x = np.array((i ,j))
    y = np.array((f, h))
    dist = np.linalg.norm(np.abs(x-y))
    from scipy.spatial import distance
    dist = distance.euclidean(y,x)
    return dist
def psnr(img1, img2):
    """
    Assuming img2 is the ground truth, we take it's PIXEL_MAX

    :param img1: Synthesized image
    :param img2: Ground Truth
    :return:
    """
    # normalize images between [0, 1]
    epsilon = 0.00001
    img2_n = img1 / (img2.max() + epsilon)
    img1_n = img2 / (img2.max() + epsilon)

    PIXEL_MAX = 1.0

    mse = np.mean((img1_n - img2_n) ** 2)
    if mse == 0:
        return 35
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr
# def psnr(img1, img2):
#     mse = np.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100

#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def Dice (im1,im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def FuzzyJaccard (im1,im2):
    A = im1.flatten()
    B = im2.flatten()
    minAB= np.minimum(A,B)
    maxAB= np.maximum(A,B)
    union= minAB.sum()/maxAB.sum()
    return union
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ##https://scikit-learn.org/0.16/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_test))
    tick_marks = np.arange(3)
    classes=['healthy', 'malign', 'Benign']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

    thresh = cm.max()/2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, cm[i,j], horizontalalignment='center', color='white' if cm[i,j] > thresh else 'black', fontsize=25, fontweight='bold')
        plt.tight_layout()
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')

def plot_confusionmatrix(epoch,RToT_model):

    Y_pred, Im_pred = RToT_model.predict(x_test)
    y_pred = np.argmax(Y_pred, axis=1)
    cm=confusion_matrix(y_test, y_pred)
    print('Classification Report')
    target_names = ['healthy', 'malign', 'Benign']
    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.figure()
    plot_confusion_matrix(cm)
    dirfile='mse/confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")


    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],decimals=2)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile='mse/Normalized_confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=30):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
