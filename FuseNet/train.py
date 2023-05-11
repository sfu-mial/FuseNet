#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:15:31 2019
This will free the gpu memory.
find running process
sudo fuser -v /dev/nvidia*
and kill the python or code PID using
sudo kill -9 "PID code here".

@author: HKhanene
"""
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import itertools
import pandas as pd
from sklearn import preprocessing
# from Network_MLP import Generator, Discriminator
from Network_transformer import Generator

from keras.utils import np_utils

#import data_in as a
# import Utilsnew as a
import Utilsnew as a
# import Utilsnew_refined as a
from Utils_model import VGG_LOSS,calculateDistance, Dice, FuzzyJaccard,psnr_torch,CustomSchedule
from keras import losses
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
# from keras.optimizers import adam

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
from sklearn.metrics import mean_squared_error
#import skimage.transform
#from skimage import data, io, filters
import numpy as np
from numpy import array
#from skimage.transform import rescale, resize
#from scipy.misc import imresize
from sklearn.metrics import jaccard_score
import skimage
print(skimage.__version__)
from  skimage.metrics import structural_similarity as ssim
import os
from sklearn.preprocessing import MinMaxScaler
import keras  as keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from keras.utils import multi_gpu_model
#import cv2
import timeit
import math
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# from skimage.measure import compare_ssim as ssim
#import skimage.filters as filter
#from skimage.filters import threshold_local

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')
logger.setLevel(logging.INFO)

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed # no need 
tf.random.set_seed(2)

image_shape =  (128,128,1)
mean_DistanceROI = []
mean_mselist = []
mean_psnrlist = []
mean_ssimlist = []
mean_Dicelist = []
mean_FJaccard = []
Tmp_ssimlist=0
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'results')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
def nean_std_data(x):
    mvec = x.mean(0)
    stdvec = x.std(axis=0) 
    return mvec, stdvec#,mvec,stdvec
def preprocess_image(x):
    return np.divide(x.astype(np.float32), 23.)

def deprocess_image(x):
    x = np.clip(x*23, 0, 23)
    return x
class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()# K.get_value(model.optimizer.lr)

    new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)
    if epoch >= 1: #and epoch % 2 == 0:
        plot_generated_images(epoch, self.model, True,Tmp_ssimlist)
        plot_confusionmatrix(epoch, self.model)
        plot_roc_curve(self.model)
        # lr schedule callback
        # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

    PlotLosses()
    if epoch == 5 or (epoch >= 10 and epoch % 10 == 0):
        self.model.save('./results/deep_spa_mse_only.h5')

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses_task = []
        self.val_losses_task = []
        self.losses_recons = []
        self.val_losses_recons = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses_recons.append(logs.get('reconstruction_output_fuse_loss'))
        self.val_losses_recons.append(logs.get('val_reconstruction_output_fuse_loss'))
        self.losses_task.append(logs.get('category_output_loss'))
        self.val_losses_task.append(logs.get('val_category_output_loss'))
        self.acc.append(logs.get('category_output_accuracy'))
        self.val_acc.append(logs.get('val_category_output_accuracy'))
        self.i += 1
        

        # clear_output(wait=True)
        plt.plot(self.x, self.losses_task, label="loss")
        plt.plot(self.x, self.val_losses_task, label="val_loss")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_task_loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("results/model_task_loss.png", bbox_inches='tight')
        plt.close("all")


        # clear_output(wait=True)
        plt.plot(self.x, self.losses_recons, label="losses_recons")
        plt.plot(self.x, self.val_losses_recons, label="val_losses_recons")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_reconst_loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("results/model_recons_loss.png", bbox_inches='tight')
        plt.close("all")
        plt.plot(self.x, self.acc, label="category_output_acc")
        plt.plot(self.x, self.val_acc, label="val_category_output_acc")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model accuracy')
        # plt.yscale('log')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig("results/model accuracy.png", bbox_inches='tight')
        plt.close("all")

plot_losses = PlotLosses()

class MyCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch >100 and K.get_value(self.beta)<=0.1:
            # K.set_value(self.beta, K.get_value(self.beta) +0.0001)
            if  K.get_value(self.alpha)<0.3:
                 K.set_value(self.alpha, K.get_value(self.alpha) +0.001)
#            K.set_value(self.alpha, max(0.75, K.get_value(self.alpha) -0.0001))
#                  K.set_value(self.beta,  min(0.7, K.get_value(self.beta) -0.0001))
        logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))

def focal_loss(gamma=5., alpha=.25): # binary only
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

# def OrthogonalProjectionLoss(features,labels, batch_size, gamma=0.5):
#         size = tf.shape(features)[0]
#         l=np.array([0,1,2,3])
#         ll= np_utils.to_categorical(l, 4)

#         lab=ll
#         print("l {.shape}".format(l))
#         for i in range (size):
#             tf.autograph.experimental.set_loop_options(
#         shape_invariants=[(lab, tf.TensorShape([None,4]))]
#     )
#             lab= K.tf.concat((lab, ll), 0)
#         print("lab {.shape}".format(lab))

#         #  features are normalized
#         features = K.tf.math.l2_normalize(features, 1)#F.normalize(features, p=2, dim=1)
#         print("features {.shape}".format(features))

#         lab = K.tf.expand_dims(lab,2) #labels[:, None]  # extend dim
#         print("labels {.shape}".format(lab))

#         mask= (K.tf.equal(lab,K.transpose (lab)))  #torch.eq(labels, labels.t()).bool().to(device)
#         print("mask {.shape}".format(mask))
#         eye= tf.cast(K.tf.eye(tf.shape(mask)[0],4), tf.bool)
#         eye_t= ~eye
#         # eye =K.tf.eye(batch_size])# torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
#         print("eye {.shape}".format(eye))
#         mask_pos = tf.cast(tf.linalg.set_diag( mask, eye_t), tf.float32)#(mask,eye, tf.zeros_like(eye))#, eye)#mask.masked_fill(eye, 0).float()
#         print("mask_pos {.shape}".format(mask_pos))
#         mask_neg = tf.cast((~mask), tf.float32)
#         print("mask_neg {.shape}".format(mask_neg))
#         dot_prod =K.dot(features, K.transpose (features))
#         print("dot_prod {.shape}".format(dot_prod))
#         # dot_prod = K.tf.expand_dims(dot_prod,2) 
#         pos_pairs_mean = K.sum(tf.linalg.matmul (mask_pos , dot_prod)) / K.sum (mask_pos + 1e-6)
#         print("pos_pairs_mean {.shape}".format(pos_pairs_mean))

#         neg_pairs_mean = K.sum(tf.matmul(mask_neg , dot_prod))/ K.sum(mask_neg + 1e-6)
#         print("neg_pairs_mean {.shape}".format(neg_pairs_mean))
#         loss = (1.0 - pos_pairs_mean) + gamma * neg_pairs_mean
#         print("loss {.shape}".format(loss))

#         return loss
def OrthogonalProjectionLoss(features,labels, batch_size, gamma=0.5):
        size = tf.shape(labels)[0]

        #  features are normalized
        features = K.tf.math.l2_normalize(features, 1)#F.normalize(features, p=2, dim=1)
        print("features {.shape}".format(features))

        labels = K.tf.expand_dims(labels,2) #labels[:, None]  # extend dim
        print("labels {.shape}".format(labels))

        mask= (K.tf.equal(labels,K.transpose (labels)))  #torch.eq(labels, labels.t()).bool().to(device)
        print("mask {.shape}".format(mask))
        eye= tf.cast(K.tf.eye(size,3), tf.bool)
        eye_t= ~eye
        # eye =K.tf.eye(batch_size])# torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        print("eye {.shape}".format(eye))
        mask_pos = tf.cast(tf.linalg.set_diag( mask, eye_t), tf.float32)#(mask,eye, tf.zeros_like(eye))#, eye)#mask.masked_fill(eye, 0).float()
        print("mask_pos {.shape}".format(mask_pos))
        mask_neg = tf.cast((~mask), tf.float32)
        print("mask_neg {.shape}".format(mask_neg))
        dot_prod =K.dot(features, K.transpose (features))
        print("dot_prod {.shape}".format(dot_prod))
        # dot_prod = K.tf.expand_dims(dot_prod,2) 
        pos_pairs_mean = K.sum(tf.linalg.matmul (mask_pos , dot_prod)) / K.sum (mask_pos + 1e-6)
        print("pos_pairs_mean {.shape}".format(pos_pairs_mean))

        neg_pairs_mean = K.sum(tf.matmul(mask_neg , dot_prod))/ K.sum(mask_neg + 1e-6)
        print("neg_pairs_mean {.shape}".format(neg_pairs_mean))
        loss = (1.0 - pos_pairs_mean) + gamma * neg_pairs_mean
        print("loss {.shape}".format(loss))

        return loss
def binary_focal_loss(gamma=2., alpha=.25):
    ##https://github.com/umbertogriffo/focal-loss-keras
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(y_true, y_pred,alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    # def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    y_true = tf.cast(y_true, tf.float32)

    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))

    # return categorical_focal_loss_fixed
def FuzzyJaccard_distance_loss(y_true, y_pred,  n_channels=1):
 	jac = 0
 	for c in range(n_channels):
            true_batch_c = y_true[:, :, :, c]
            pred_batch_c = y_pred[:, :, :, c]
            intersect_batch = (K.minimum(true_batch_c, pred_batch_c))
            union_batch = (K.maximum(true_batch_c, pred_batch_c))
            intersect = K.sum(K.sum(intersect_batch, axis=-1), axis=-1)
            union = K.sum(K.sum(union_batch, axis=-1), axis=-1)
            j = intersect / union
            jac += K.mean(j)
 	union= jac / n_channels
 	union= (-K.log(K.clip(union, K.epsilon(), None) )) #**0.3
 	return union # (1- union )

#def centroideloss(y_true, y_pred):
#    true_batch_c = y_true[:, :, :, 1]
#    pred_batch_c = y_pred[:, :, :, 1]
#
#    #mask image by thresholding over the mean
#    true_batch_thre = y_true[y_true>K.meam(y_true)]
#    pred_batch_thre = y_pred[y_pred>K.meam(y_pred)]
#    # find min and max per image
#    minx_pred_batch_thre= K.min(pred_batch_thre , axis=0)
#    maxx_pred_batch_thre= K.max(pred_batch_thre , axis=0)
#    miny_pred_batch_thre= K.min(pred_batch_thre , axis=1)
#    maxy_pred_batch_thre= K.max(pred_batch_thre , axis=1)
#
#    minx_true_batch_thre= K.min(true_batch_thre , axis=0)
#    maxx_true_batch_thre= K.max(true_batch_thre , axis=0)
#    miny_true_batch_thre= K.min(true_batch_thre , axis=1)
#    maxy_true_batch_thre= K.max(true_batch_thre , axis=1)
#
#    centroid_pred= [(maxx_pred_batch_thre-minx_pred_batch_thre)/2. , (maxy_pred_batch_thre-miny_pred_batch_thre)/2.]
#    centroid_true= [(maxx_true_batch_thre-minx_true_batch_thre)/2. , (maxy_true_batch_thre-miny_true_batch_thre)/2.]
#    return losses.mean_squared_error(centroid_pred, centroid_true)
#
def grad( y, x ):
    return Lambda( lambda z: K.gradients( z[ 0 ], z[ 1 ] ), output_shape = [0] )( [ y, x ] )

def second_derivative(y_true, y_pred):   
    loss=0
    size= K.int_shape(y_true)[0]
    if (size is None):
        size= 1
        print("here")
    for i in range(size):
        
        true_batch_c = K.tf.squeeze(y_true[i, :, :,:])
        pred_batch_c = K.tf.squeeze(y_pred[i, :, :,:])

        L1_Distance= K.mean(K.sum(K.abs(laplacian_of_gaussian(true_batch_c, 5,1) - laplacian_of_gaussian(pred_batch_c, 5,1)), axis=-1), axis=-1)/(128)      #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))
        loss += L1_Distance
    return loss/ size

def laplacian(image, size):
#    if len(image.shape)==3 and image.shape[2]==3: # convert rgb to grayscale
#        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
#        image = tf.image.rgb_to_grayscale(image)
#        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
#        image = tf.squeeze(image,2)
#    
#    image = K.tf.convert_to_tensor(image, dtype=tf.float32)

#    new = K.tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = K.tf.expand_dims(image,2)
    image = K.tf.expand_dims(new,0)
    
    fil = np.ones([size, size])
    fil[int(size/2), int(size/2)] = 1.0 - size**2
    fil = K.tf.convert_to_tensor(fil, tf.float32)
    fil = K.tf.stack([fil]*1, axis=2)
    fil = K.tf.expand_dims(fil, 3)
    print("fil {.shape}".format(fil))
    print("image {.shape}".format(image))


    result = K.tf.nn.depthwise_conv2d(image, fil, strides=[1, 1, 1, 1], padding="SAME")
    result = K.tf.squeeze(result,0)
    result = K.tf.squeeze(result,2)
    
#    result = result.numpy()
    minM = K.min(result,axis=-1) # np.min(result)
    maxM = K.max(result,axis=-1)#np.max(result)
    output = (result - minM) / (maxM - minM)
    return output

def laplacian_of_gaussian(image, filtersize, sigma):

    n_channels = 1
    image = K.tf.expand_dims(image, 2)
    
    w = math.ceil(sigma * filtersize)
    w_range = int(math.floor(w/2))

    y = x = K.tf.range(-w_range, w_range+1, 1)
    Y, X = K.tf.meshgrid(x, y)
    z = K.tf.cast(tf.add(tf.square(X), tf.square(Y)),tf.float32)
    nom = K.tf.subtract(z, 2*(sigma**2))
    denom = 2*math.pi*(sigma**6)
    exp = K.tf.exp( -z/2*(sigma**2))
    fil = K.tf.divide(tf.multiply(nom, exp), denom)
    
    fil =K. tf.stack([fil]*n_channels, axis=2)
    fil = K.tf.expand_dims(fil, 3)
    
#    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = K.tf.expand_dims(image, 0)
    res = K.tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")
    res = K.tf.squeeze(res,0)
    res = K.tf.squeeze(res,2)
    
#    result = res.numpy()
    minM = K.min(res,axis=-1) #np.min(result)
    maxM = K.max(res,axis=-1)#np.max(result)
    output = (res- minM) / (maxM - minM)
    return output
        
def dst_transform(y_true, y_pred):   
    loss=0
    size= K.int_shape(y_true)[0]
    if (size is None):
        size= 1
        print("here")
    for i in range(size):
#         print("y_true {.shape}".format(y_true))

        true_batch_c = K.tf.squeeze(y_true[i, :, :,:])
        pred_batch_c = K.tf.squeeze(y_pred[i, :, :,:])
#         print("true_batch_c {.shape}".format(true_batch_c))

        # ###### dist_transform for y_true#####
        y_true_mask_f = K.cast(K.greater(true_batch_c,(K.mean(true_batch_c)+K.std(true_batch_c))), dtype='float32')
#         print("y_true_mask_f {.shape}".format(y_true_mask_f))

        ones_true = K.tf.where(K.tf.greater_equal(y_true_mask_f,1))
#         print("ones_true {.shape}".format(ones_true))
        zeros_true = K.tf.where(K.tf.equal(y_true_mask_f,0))
#         print("zeros_true {.shape}".format(zeros_true))

        X_true = K.cast(ones_true,dtype="float32")
        Y_true =  K.cast(zeros_true,dtype="float32")
        is_empty_gt = K.tf.equal(K.tf.size(X_true), 0)
        a_true = -2*K.dot( Y_true, K.transpose(X_true))
        b_true  = K.sum(K.square(X_true), axis=1) 
        c_true  =K.tf.expand_dims(K.sum(K.square(Y_true), axis=1), 1)
        sumall_true =a_true +b_true +c_true 
#     return sumall_true#losss/batch_size #K.mean(finalChamferDistanceSum)

        dists_true = K.sqrt(K.min(sumall_true,axis=-1)) # min dist of each zero pixel to one pixel
#     print("dists {.shape}".format(dists_true))
        dist_trans_true= K.zeros ((128,128),dtype='float32')
        shape= (128,128)
        delta_true = K.tf.SparseTensor(zeros_true, dists_true, shape)
        result_t = dist_trans_true + K.tf.sparse.to_dense(delta_true)
        result_true = K.tf.cond(is_empty_gt, lambda: dist_trans_true, lambda: result_t/128)
    
# ###### dist_transform for y_pred#####
#     y_pred_mask_f = K.cast(K.greater_equal(y_pred_f,K.mean(y_pred)), dtype='float64')
        y_pred_mask_f = K.cast(K.greater(pred_batch_c,(K.mean(pred_batch_c)+K.std(pred_batch_c))), dtype='float32')
        ones_pred= K.tf.where(K.tf.greater_equal(y_pred_mask_f,1))
#     print("ones_true {.shape}".format(ones_true))
        zeros_pred= K.tf.where(K.tf.equal(y_pred_mask_f,0))
#     print("zeros_true {.shape}".format(zeros_true))

        X_pred = K.cast(ones_pred,dtype="float32")
        Y_pred =  K.cast(zeros_pred,dtype="float32")
        is_empty_pre = K.tf.equal(K.tf.size(X_pred), 0)
    
        a_pred = -2*K.dot( Y_pred, K.transpose(X_pred))
        b_pred  = K.sum(K.square(X_pred), axis=1) 
        c_pred  =K.tf.expand_dims(K.sum(K.square(Y_pred), axis=1), 1)
        sumall_pred =a_pred +b_pred +c_pred
        dists_pred = K.sqrt(K.min(sumall_pred,axis=-1)) # min dist of each zero pixel to one pixel
#     print("dists {.shape}".format(dists_true))
        dist_trans_pred= K.zeros ((128,128),dtype='float32')
        shape= (128,128)
        delta_pred = K.tf.SparseTensor(zeros_pred, dists_pred, shape)
        result_p = dist_trans_pred + K.tf.sparse.to_dense(delta_pred)
        result_pred = K.tf.cond(is_empty_pre, lambda: dist_trans_pred, lambda: result_p/128)
    
#        L2_Distance= K.sum(K.mean(K.square(result_true - result_pred), axis=-1), axis=-1)/(128*128*2)
#        L2_Distance= K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))       #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))

#        L1_Distance= K.mean(K.sum(K.abs(K.maximum(K.cast(0,dtype="float32"),result_true - result_pred)), axis=-1), axis=-1)/(128)      #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))
        L1_Distance= K.mean(K.sum(K.abs(result_true - result_pred), axis=-1), axis=-1)/(128)      #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))

        loss += L1_Distance
    return loss/ size

def loss(alpha, Beta,batch_size,feature, gamma):
    def custom_loss_func(y_true, y_pred):
        return custom_loss(y_true, y_pred, alpha, Beta,batch_size,feature,gamma)
    return custom_loss_func


# def custom_loss(y_true, y_pred):
# #    alpha=0.999995
#     loss =1*losses.mean_squared_error(y_true, y_pred)# was 1*   mean_squared_error
#     loss+= 0.3*FuzzyJaccard_distance_loss(y_true, y_pred)# was 0.4, 0.3
#     return  loss
def weighted_categorical_crossentropy(y_true, y_pred,weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    # def loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    # return loss
    
    return loss
def custom_loss(y_true, y_pred, alpha, Beta,batch_size,feature, gamma):
    # loss =losses.mean_squared_error(y_true, y_pred)# was 1*   mean_squared_error

    # loss+= Beta*FuzzyJaccard_distance_loss(y_true, y_pred)#  Beta* was 0.4, 0.3
    # loss +=alpha* dst_transform(y_true, y_pred)
    weights= np.array([0.3,.5,.5])
    ce =losses.categorical_crossentropy(y_true, y_pred)
    loss= weighted_categorical_crossentropy(y_true, y_pred,weights)
    # loss=categorical_focal_loss(y_true, y_pred, alpha=[[.25, .25, .25]], gamma=2.)
    loss +=0.3*OrthogonalProjectionLoss(feature ,y_true, batch_size, gamma=0.7) 
    return  loss/2

def normalize_data(values):
    from math import sqrt

    # epsilon=0.001
    # mvec = x.mean(1)
    # stdvec = x.std(axis=1)
    # return ((x - mvec)/stdvec)# s,mvec,stdvec
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    # print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


def minmax_normalization(x):
	scaler = MinMaxScaler()

	MinMax = scaler.fit(x)
	MinMax_XEAS= scaler.transform(x)

	return MinMax_XEAS , MinMax

def minmax_normalization_ref(x,scaler):
#	MinMax = scaler.fit(x)
	MinMax_XEAS= scaler.transform(x)
	return MinMax_XEAS


def total_variation_loss(x):
    assert 4 == K.ndim(x)

    img_nrows=128
    img_ncols=128
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, 1:, :img_ncols-1])
        b = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, :img_nrows-1, 1:])
    else:
        a = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
        b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    loss= K.sum(K.pow(a + b, 1.25))
    return loss

def normalized_intensity(actualscan):

    max_actuali = np.max(actualscan[:,0:128], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency
    max_actualii = np.max(actualscan[:,128:], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency

    min_actuali = np.min(actualscan[:,0:128], axis=1); ##min value of each row
    min_actualii = np.min(actualscan[:,128:], axis=1); ##min value of each row


    minmaxi=(max_actuali - min_actuali).reshape((actualscan[:,0:128] .shape[0],1))
    minmaxii=(max_actualii - min_actualii).reshape((actualscan[:,128:].shape[0],1))


    normalized_intensityi =np.divide((actualscan[:,0:128] - min_actuali.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxi)
    normalized_intensityii =np.divide((actualscan[:,128:] - min_actualii.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxii)
    #    normalized_intensity= normalized_intensityi+ normalized_intensityii
    normalized_intensity= np.concatenate((normalized_intensityi, normalized_intensityii), axis=1)  		#normalized_intensityi+ normalized_intensityii


    return normalized_intensity

def vgg_loss(y_true, y_pred):

#    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
#    vgg19.trainable = False
#    for l in vgg19.layers:
#        l.trainable = False
#    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
#    loss_model.trainable = False
#    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
    vgg19= load_model('./autoencoder/deep_spa_l2f3x3l90.h5')# Pretrained model on SA and JK

#    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_network(shape, generator, optimizer):
    gan_input = Input(shape=(1024,))
    x = generator(gan_input)
    gan = Model(inputs=gan_input, outputs=x)
#    gan.compile(loss='mse', # customerloss# vgg_loss
                #loss_weights=[1, 1e-3 ],# 1, 1e-3
#                optimizer=optimizer)

# def plot_generated_images(epoch,generator, val =True, examples=5, dim=(1, 2), figsize=(10, 5)):
#     DistanceROI = []
#     mselist=[]
#     psnrlist=[]
#     ssimlist=[]
#     Dicelist= []
#     FJaccard=[]
#     label=[]
#     GT_label=[]
#     vmin=0
#     vmax=25
#     global Tmp_ssimlist
#     if (val ==True):
#         # rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
#         image_batch_hr = x_test_hr
#         image_batch_lr = x_test_lr#0:25
#         img =y_testima[:,:]
#         examples=len(x_test_hr)#
#         dirfile='results/test_generated_image_epoch_'
    
#     generated_image = generator.predict(image_batch_lr)
# #    generated_image = denormalize(gen_img)
# #    image_batch_lr = denormalize(image_batch_lr)

#     #generated_image = deprocess_HR(generator.predict(image_batch_lr))
#     for index in range(examples):
#             if (val==False):
#              image_batch_hr[index]=(image_batch_hr[index]).reshape(128, 128)
#             fig=plt.figure(figsize=figsize)

#             ax1=plt.subplot(dim[0], dim[1], 1)
#             imgn = img[index] #/ np.linalg.norm(image_batch_hr[index])
#             im1 = ax1.imshow(imgn.reshape(128, 128))  # , interpolation='nearest')
#             divider = make_axes_locatable(ax1)
#             cax = divider.append_axes('right', size='5%', pad=0.05)
#             fig.colorbar(im1, cax=cax, orientation='vertical')
#             label.append(generated_image)
#             GT_label.append(x_test_hr)

#             plt.close("all")

#     loss_file = open('results/losses.txt' , 'a')
#     if (val == True):
#         loss_file.write('synthe  epoch%d :  GT_label = %s ; label = %s  \n' %(epoch, GT_label, label ) )
#     # if Tmp_ssimlist<ssim_mean:
#     #     generator.save('./mse/ModelCheckpoint.h5')
#     #     print('ssim improved from %s to %s, saving model to weight\n' %(Tmp_ssimlist, ssim_mean))
#     #     Tmp_ssimlist = ssim_mean
def plot_generated_images(epoch,generator, val =True, examples=5, dim=(1, 6), figsize=(10, 5)):
    fg_color = 'black'
    bg_color =  'white'
    DistanceROI = []
    mselist=[]
    psnrlist=[]
    ssimlist=[]
    Dicelist= []
    FJaccard=[]
    vmin=0
    vmax=25
    PD_label=[]
    GT_label=[]
    global Tmp_ssimlist
    if (val ==True):
        # rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
        image_batch_hr = x_test_hr[:,:]
        image_batch_lr1 = x_test_lr_1[:,:]#0:25
        image_batch_lr2 = x_test_lr_2[:,:]#0:25
        image_batch_lr3 = x_test_lr_3[:,:]#0:25
        image_batch_lr4 = x_test_lr_4[:,:]#0:25

        examples=len(x_test_hr)#
        dirfile='results/test_generated_image_epoch_'
    
    label, generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f = generator.predict([image_batch_lr1,image_batch_lr2,image_batch_lr3,image_batch_lr4])
#    generated_image = denormalize(gen_img)
#    image_batch_lr = denormalize(image_batch_lr)

    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    for index in range(examples):
            if (val==False):
             image_batch_hr[index]=(image_batch_hr[index]).reshape(128, 128)
            fig=plt.figure(figsize=figsize)
            #combined_data = np.array([ image_batch_hr[index], generated_image[index].reshape(128, 128) ])
            #_min, _max = np.amin(combined_data), np.amax(combined_data)
            # _vmin1 = (image_batch_hr[index]).min()
            # _vmax1 = (image_batch_hr[index]).max()
            #
            # _vmin2 = (generated_image[index]).min()
            # _vmax2 = (generated_image[index]).max()
            # vmin= min(vmin , min(_vmin1, _vmin2))
            # vmax = max(vmax, min(_vmax1, _vmax2))

#        	ax1=plt.subplot(dim[0], dim[1], 1)
            ax1=plt.subplot(dim[0], dim[1], 1)
            ax1.set_title('GT', color=fg_color)
            imgn = np.flipud(image_batch_hr[index]) #/ np.linalg.norm(image_batch_hr[index])
            im1 = ax1.imshow(imgn.reshape(128, 128))  # , interpolation='nearest')
            # im1=ax1.imshow(image_batch_hr[index].reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.axis('off')
            fig.colorbar(im1, cax=cax, orientation='vertical')



            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image_1[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax3=plt.subplot(dim[0], dim[1], 3)
            imgnr = np.flipud(generated_image_2[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax3.set_title('Recons_f2', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax4=plt.subplot(dim[0], dim[1], 4)
            imgnr = np.flipud(generated_image_3[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax4.set_title('Recons_f3', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax4.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax5=plt.subplot(dim[0], dim[1], 5)
            imgnr = np.flipud(generated_image_4[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax5.set_title('Recons_f4', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax5.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax6=plt.subplot(dim[0], dim[1], 6)
            imgnr = np.flipud(generated_image_f[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax6.set_title('Recons_all', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax6.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')


            plt.tight_layout(pad=0.01)
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
            #a=image_batch_hr[index]

#            a=cv2.normalize( image_batch_hr[index],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #b=generated_image[index]#
#            b=cv2.normalize( generated_image[index],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            v=calculateDistance (generated_image_f[index],image_batch_hr[index])#
            DistanceROI.append(v)
#            mse=mean_squared_error(a,b)
#            mselist.append(mse)
            p=psnr_torch(generated_image_f[index],image_batch_hr[index])#(y_te[i],decoded_imgs[i].reshape(128, 128)) #(a.reshape(128, 128),b)#
            psnrlist.append(p)
            ss_im = ssim(image_batch_hr[index].reshape(128, 128), generated_image_f[index].reshape(128, 128))
            ssimlist.append(ss_im)
#        	if min(image_batch_hr[index])==max(generated_image[index]):
#        		threshold_GT = filter.threshold_otsu(a)
#        		threshold_Rec = filter.threshold_otsu(b)
#    #    plt.imshow(a > threshold_GT )
#        		gt=a > threshold_GT
#
#        		seg=b > 0.6
            #dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
#        		dices= Dice(gt,seg)
#        		dices=jaccard_similarity_score(gt,seg)
#        		Dicelist.append(dices)
            fjacc= FuzzyJaccard(image_batch_hr[index],generated_image_f[index])
            FJaccard.append(fjacc)
            plt.close("all")
            PD_label.append(label[index])
            GT_label.append(y_testlabel[index])
    FJ_mean= np.mean(FJaccard)
    FJ_std= np.std(FJaccard)
    DistanceROI_mean= np.mean(DistanceROI)
    DistanceROI_std= np.std(DistanceROI)
#    mse_mean=np.mean(mselist)
#    mse_std=np.std(mselist)
    psnr_mean=np.mean(psnrlist)
    psnr_std=np.std(psnrlist)
    ssim_mean=np.mean(ssimlist)
    ssim_std=np.std(ssimlist)
#    dice_mean=np.mean(Dicelist)
#    dice_std=np.std(Dicelist)
    loss_file = open('results/losses.txt' , 'a')
    if (val == True):
        loss_file.write('synthe  epoch%d :  DistanceROI = %s + ~  %s ; psnr_mean = %s + ~  %s ; ssim_mean = %s + ~  %s ; FuzzyJaccard_mean = %s + ~ %s \n' %(epoch, DistanceROI_mean, DistanceROI_std,psnr_mean,psnr_std,ssim_mean, ssim_std, FJ_mean, FJ_std ) )
    if Tmp_ssimlist<ssim_mean:
        generator.save('./results/Checkpoint.h5')
        print('ssim improved from %s to %s, saving model to weight\n' %(Tmp_ssimlist, ssim_mean))
        Tmp_ssimlist = ssim_mean
    loss_file = open('results/losses_acc.txt' , 'a')
    if (val == True):
        loss_file.write('synthe  epoch%d :  GT_label = %s ; label = %s  \n' %(epoch, GT_label, label ) )


#    loss_file.write('epoch%d : DistanceROI_mean = %s + ~ %s ;  psnr_mean = %s + ~  %s  ; FuzzyJaccard_mean = %s + ~ %s \n' %(epoch, DistanceROI_mean,DistanceROI_std, psnr_mean,psnr_std, FJ_mean, FJ_std ) )
    loss_file.close()



#    loss_file.write('epoch%d : DistanceROI_mean = %s + ~ %s ;  psnr_mean = %s + ~  %s  ; FuzzyJaccard_mean = %s + ~ %s \n' %(epoch, DistanceROI_mean,DistanceROI_std, psnr_mean,psnr_std, FJ_mean, FJ_std ) )
    loss_file.close()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ##https://scikit-learn.org/0.16/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_test))
    tick_marks = np.arange(3)
    classes=['healthy', 'Benign', 'malign']
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

    #  Y_pred = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    
    # y_pred = np.argmax(Y_pred, axis=1)
    # cm=confusion_matrix(y_test, y_pred)
    Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    y_pred = np.argmax(Y_pred, axis=1)
    # print(Y_pred)
    # print(y_pred)
    # lb = preprocessing.LabelBinarizer()
    # print(lb.transform(y_pred))
    # print(y_testlabel)
    cm=confusion_matrix(y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['healthy', 'Benign', 'malign']
    print(y_testlabel.shape, y_pred.shape)
    print(classification_report(y_testlabel, y_pred, target_names=target_names))
    Classification_Report_file = open('results/losses.txt' , 'a')
    report = classification_report(y_testlabel, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('results/Classification_Report_file.txt',  header=True, index=False, sep='\t', mode='a')
    # Classification_Report_file.write('synthe  epoch%d :  classification_report%s \n' %(epoch, df))
    Classification_Report_file.close()


    plt.figure()
    plot_confusion_matrix(cm)
    dirfile='results/confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")


    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],decimals=2)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile='results/Normalized_confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

def plot_roc_curve(model):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    # Binarize the output
    y = label_binarize(y_testlabel, classes=[0, 1, 2])
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_testlabel))]
    # keep probabilities for the positive outcome only
    # Y_pred = model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    lr_probs_ben = Y_pred[:, 1]
    lr_probs_mal = Y_pred[:, 2]
    lr_probs_heal = Y_pred[:, 0]

    # calculate scores
    lr_auc_ben = roc_auc_score(y[:, 1], lr_probs_ben)
    ns_auc = roc_auc_score(y[:, 1], ns_probs)
    lr_auc_mal = roc_auc_score(y[:, 2], lr_probs_mal)
    lr_auc_H = roc_auc_score(y[:, 0], lr_probs_heal)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Benign: ROC AUC=%.3f' % (lr_auc_ben))
    print('Malignant: ROC AUC=%.3f' % (lr_auc_mal))
    print('Healthy: ROC AUC=%.3f' % (lr_auc_H))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y[:, 1], ns_probs)
    lr_fprb, lr_tprb, _ = roc_curve(y[:, 1], lr_probs_ben)
    lr_fprm, lr_tprm, _ = roc_curve(y[:, 2], lr_probs_mal)
    lr_fprh, lr_tprh, _ = roc_curve(y[:, 0], lr_probs_heal)


    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='')
    plt.plot(lr_fprb, lr_tprb, marker='.', label='Benign(ROC AUC=%.3f)' % (lr_auc_ben))
    plt.plot(lr_fprm, lr_tprm, marker='+', label='Malignant(ROC AUC=%.3f)' % (lr_auc_mal))
    plt.plot(lr_fprh, lr_tprh, marker='x', label='Healthy (ROC AUC=%.3f)' % (lr_auc_H))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    dirfile='results/Roc_curve'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

def loss_r(alpha, Beta,batch_size):
    def custom_loss_func_r(y_true, y_pred):
        return custom_loss_r(y_true, y_pred, alpha, Beta,batch_size)
    return custom_loss_func_r

def custom_loss_r(y_true, y_pred, alpha, Beta,batch_size):  # orthogonal_loss
    loss =losses.mean_squared_error(y_true, y_pred)# was 1*   mean_squared_error
#    loss +=losses.mean_absolute_error(y_true, y_pred)# was 1*   mean_squared_error

    loss+= Beta*FuzzyJaccard_distance_loss(y_true, y_pred)#  Beta* was 0.4, 0.3
    loss +=alpha* dst_transform(y_true, y_pred)
#    loss +=second_derivative(y_true, y_pred)#dst_transform(y_true, y_pred)
    # loss+= 0.3*vgg_loss(y_true, y_pred)
    return  loss


def train(epochs=1, batch_size=64):
    alpha = K.variable(0.2)#(0.1)
    beta = K.variable(0.25)#(0.02)
    gamma = K.variable(0.5)#(0.02)
    global Tmp_ssimlist
    Tmp_ssimlist = 0

    shape = (256,)
    get_feat=1
    feature, feature_cla, generator = Generator(shape).generator()
#    discriminator = Discriminator(image_shape).discriminator()
    # sgd = SGD(lr=0.001, momentum=0.9, decay=1e-4)# momentum=0.8, decay=1e-6)
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)#,clipvalue=1.0) # was r=1E-5 here
#    Nadam=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    model_loss_fuse= loss(alpha, beta,batch_size,feature_cla,gamma)#feature_cla
    model_loss= loss_r(alpha, beta,batch_size) 
#    model=generator
#     generator= load_model('SA_only/deep_spa_l2f3x3l200.h5')# Pretrained model on SA and JK
# #
#    model = multi_gpu_model(generator, gpus=4)
    model=generator
    model.compile(    loss = {
        "category_output":
         model_loss_fuse,
        #  [categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2.)],
        # tf.keras.losses.CategoricalCrossentropy(),
        "reconstruction_output1": model_loss,#tf.keras.losses.MeanSquaredError(),# model_loss,#
        "reconstruction_output2": model_loss,#tf.keras.losses.MeanSquaredError(),#
        "reconstruction_output3":  model_loss,#tf.keras.losses.MeanSquaredError(),
        "reconstruction_output4":  model_loss,#tf.keras.losses.MeanSquaredError(),
        "reconstruction_output_fuse": model_loss,#tf.keras.losses.MeanSquaredError()#model_loss#
        #loss_weights={'age_output': .001, 'gender_output': 1.})

    },
    metrics = {
        # "category_output": 'accuracy',
        "reconstruction_output_fuse": 'mse'
    }

    , optimizer=tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)) #'RMSprop'
    # layer_name = 'fusion'
    # feature = model.layers[21].output
    # model(inputs=model.input,outputs=model.get_layer(layer_name).output())
    loss_file = open('results/losses.txt' , 'w+')
    # Classification_Report_file  = open('results/Classification_Report.txt' , 'w+')

    loss_file.close()
    # Classification_Report_file.close()

    keras.callbacks.Callback()
# lr decay function
    def lr_decay(epoch):
        return 0.1 * math.pow(0.1, epoch)
    # lr decay function 2
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=0, mode='auto') #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
# lr schedule callback
    # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

    def scheduler(epoch):
        if epoch >= 1: #and epoch % 2 == 0:
            plot_generated_images(epoch, model, True,Tmp_ssimlist)
            plot_confusionmatrix(epoch, model)
            plot_roc_curve(model)
            # lr schedule callback
            # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

            PlotLosses()
            if epoch == 5 or (epoch >= 10 and epoch % 10 == 0):
                generator.save('./results/deep_spa_mse_only.h5')


        return K.get_value(model.optimizer.lr)

    # for iteration in (0, 100):
    change_lr = LearningRateScheduler(scheduler)
    from keras.callbacks import ModelCheckpoint
    filepath="results/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1,  monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

    history =model.fit([x_train_lr_1, x_train_lr_2, x_train_lr_3, x_train_lr_4], [ y_label,x_train_hr, x_train_hr, x_train_hr, x_train_hr, x_train_hr], epochs=epochs, batch_size=batch_size, shuffle=True,
    validation_split=0.1,
    # validation_data=(x_test_lr,x_test_hr),
     callbacks = [MyCallback(alpha, beta), plot_losses,checkpoint,LearningRateReducerCb(),reduce_lr])#,lr_decay_callback,change_lr

def normalize_test(x,m):
    epsilon=0.01
    mvec = m.mean(1)
    #print(mvec)
    stdvec = m.std(axis=1)
    #print(stdvec)
    return ((x - mvec)/stdvec+epsilon)# s,mvec,stdvec
measure_1,measure_2,measure_3,measure_4, immatrix, label, testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,immatrix_test ,label_test=a.load_data()
#old  x_train, y_train, y_label, x_test, y_test, y_testlabel= a.load_data()

x_train_1= measure_1+ np.random.normal( measure_1.mean()/3, measure_1.mean()/2, 256)
x_train_2= measure_2+ np.random.normal( measure_2.mean()/3, measure_2.mean()/2, 256)
x_train_3= measure_3+ np.random.normal( measure_3.mean()/3, measure_3.mean()/2, 256)
x_train_4= measure_4+ np.random.normal( measure_4.mean()/3, measure_4.mean()/2, 256)

y_train= immatrix
x_test_2= testmeasure_2+ np.random.normal( testmeasure_2.mean()/3, testmeasure_2.mean()/2, 256)
x_test_1= testmeasure_1+ np.random.normal( testmeasure_1.mean()/3, testmeasure_1.mean()/2, 256)
x_test_3= testmeasure_3+ np.random.normal( testmeasure_3.mean()/3, testmeasure_3.mean()/2, 256)
x_test_4= testmeasure_4+ np.random.normal( testmeasure_4.mean()/3, testmeasure_4.mean()/2, 256)
# x_train= measure_750
y_train_label= label
y_trainima= immatrix


x_train_1= normalize_data (x_train_1) #here
x_test_1 = normalize_data(x_test_1) #here

x_train_2= normalize_data (x_train_2) #here
x_test_2= normalize_data(x_test_2) #here

x_train_3= normalize_data (x_train_3) #here
x_test_3 = normalize_data(x_test_3) #here

x_train_4= normalize_data (x_train_4) #here
x_test_4= normalize_data(x_test_4) #here
# x_test= testmeasure_750
y_testlabel=label_test
y_testima= immatrix_test

x_train_lr_2 = x_train_2
x_train_lr_1 = x_train_1
x_train_lr_3 = x_train_3
x_train_lr_4 = x_train_4

x_test_lr_2 = x_test_2
x_test_lr_1 = x_test_1
x_test_lr_4 = x_test_4
x_test_lr_3 = x_test_3

y_train = np.reshape(y_trainima, (len(y_trainima), 128, 128,1))  #

y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  #
x_train_hr=y_train
x_test_hr=  y_test

print("data loaded")


#x_real = normalize_data(x_real)

# y_train = np.reshape(y_train, (len(y_train), 128, 128,1))  #

# y_testima = np.reshape(y_test, (len(y_test), 128, 128,1))  #


from sklearn.preprocessing import label_binarize
# Binarize the output
# y = label_binarize(Y_testlabel, classes=[0, 1, 2])
#test images
# x_train_lr = x_train
y_label= label_binarize(y_train_label, classes=[0, 1, 2])#np_utils.to_categorical(y_train, 3)


# x_test_lr = x_test
Y_testlabel=label_binarize(y_testlabel, classes=[0, 1, 2])# #np_utils.to_categorical(y_test, 3)
Y_testlabel=np_utils.to_categorical(y_testlabel, 3)




print("data processed")

train(25,16)



# was 32

