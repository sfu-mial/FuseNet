#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:15:31 2019

@author: HKhanene
"""
import numpy as np
from numpy import array
from sklearn.preprocessing import label_binarize
import random
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

#import data_in as a
import Utilsnew as a
import Utilsnew_Clinicalset as a_clin
import Utils_TL_new as a_TL
# import Utilsnew_testset as a
#import Utils_old as a
from Utils_model import VGG_LOSS,calculateDistance, Dice, FuzzyJaccard,psnr_torch,plot_confusion_matrix,plot_confusionmatrix
from keras import losses
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
from sklearn.metrics import mean_squared_error
#import skimage.transform
#from skimage import data, io, filters

#from skimage.transform import rescale, resize
#from scipy.misc import imresize
import os
from sklearn.preprocessing import MinMaxScaler
import keras  as keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import cv2
import timeit
import math
from sklearn.metrics import jaccard_score
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score 
from sklearn.preprocessing import label_binarize

from  skimage.metrics import structural_similarity as ssim
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
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
patien_id=1


print ("patient id", patien_id)

def extract_csv_gen_plot(csv_path):

    data = pd.read_csv(csv_path, index_col=1)
    data = data.drop(data.columns[[0, 1]], axis=1)
    data.index.names = ['Name']
    g = sns.heatmap(data)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    g.set_title('Heatmap')
    plt.tight_layout()
    plt.show()


current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'testcase_mmc')
Clinical_directory = os.path.join(current_directory, 'Clinical_GNonly-ad/Patient'+str(patien_id))
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
if not os.path.exists(Clinical_directory):
   os.makedirs(Clinical_directory)

def nean_std_data(x):
    mvec = x.mean(0)
    stdvec = x.std(axis=0) 
    return mvec, stdvec#,mvec,stdvec
def preprocess_image(x):
    return np.divide(x.astype(np.float32), 23.)

def deprocess_image(x):
    x = np.clip(x*23, 0, 23)
    return x

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
        self.losses_recons.append(logs.get('reconstruction_output_loss'))
        self.val_losses_recons.append(logs.get('val_reconstruction_output_loss'))
        self.losses_task.append(logs.get('category_output_loss'))
        self.val_losses_task.append(logs.get('val_category_output_loss'))
        self.acc.append(logs.get('category_output_acc'))
        self.val_acc.append(logs.get('val_category_output_acc'))
        self.i += 1
        

        # clear_output(wait=True)
        plt.plot(self.x, self.losses_task, label="loss")
        plt.plot(self.x, self.val_losses_task, label="val_loss")
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_task_loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("testcase_mmc/model_task_loss.png", bbox_inches='tight')
        plt.close("all")


        # clear_output(wait=True)
        plt.plot(self.x, self.losses_recons, label="losses_recons")
        plt.plot(self.x, self.val_losses_recons, label="val_losses_recons")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_reconst_loss')
        plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("testcase_mmc/model_recons_loss.png", bbox_inches='tight')
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
        plt.savefig("testcase_mmc/model accuracy.png", bbox_inches='tight')
        plt.close("all")

plot_losses = PlotLosses()

def normalized_intensity(actualscan):
    
    max_actuali = np.max(actualscan[:,0:128], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency
    max_actualii = np.max(actualscan[:,128:], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency
    
    min_actuali = np.min(actualscan[:,0:128], axis=1); ##min value of each row
    min_actualii = np.min(actualscan[:,128:], axis=1); ##min value of each row
    
    minmaxi=(max_actuali - min_actuali).reshape((actualscan[:,0:128] .shape[0],1))
    minmaxii=(max_actualii - min_actualii).reshape((actualscan[:,128:].shape[0],1))

    normalized_intensityi =np.divide((actualscan[:,0:128] - min_actuali.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxi)
    normalized_intensityii =np.divide((actualscan[:,128:] - min_actualii.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxii)
    normalized_intensity= np.concatenate((normalized_intensityi, normalized_intensityii), axis=1)  #normalized_intensityi+ normalized_intensityii

    return normalized_intensity
def normalize_data(values):
    from sklearn.preprocessing import StandardScaler
    # epsilon=0.001
    # mvec = x.mean(0)
    # stdvec = x.std(axis=0)
    # return ((x - mvec)/stdvec)# s,mvec,stdvec

    """get a distribution mean and std
    """
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    # print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized
def normalize_data_ref(values_tr,values):
    from sklearn.preprocessing import StandardScaler
    # epsilon=0.001
    # mvec = x.mean(0)
    # stdvec = x.std(axis=0)
    # return ((x - mvec)/stdvec)# s,mvec,stdvec

    """get a distribution mean and std
    """
    scaler = StandardScaler()
    scaler = scaler.fit(values_tr)
    # print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


def plot_generated_images_clini(dirfile,generator,examples=5, dim=(1, 6), figsize=(10, 5)):
    fg_color = 'black'
    bg_color =  'white'
    vmin=0
    vmax=25
    PD_label=[]
    GT_label=[]
    global Tmp_ssimlist
 
    dirfile='Clinical_GNonly-ad/Patient'+str(patien_id)+'/Clinical_predicted_image_'
    
    label,generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f = generator.predict([x_test_lr_1,x_test_lr_2,x_test_lr_3,x_test_lr_4])
    print (generated_image_1.shape)
#    generated_image = denormalize(gen_img)
#    image_batch_lr = denormalize(image_batch_lr)
    v_min=0.34

    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    for index in range(len(generated_image_1)):
            fig=plt.figure(figsize=figsize)

            # if (val==False):
            #  image_batch_hr[index]=(image_batch_hr[index]).reshape(128, 128)
            # ax1=plt.subplot(dim[0], dim[1], 1)
            # ax1.set_title('GT', color=fg_color)
            # imgn = np.flipud(image_batch_hr[index]) #/ np.linalg.norm(image_batch_hr[index])
            # im1 = ax1.imshow(imgn.reshape(128, 128))  # , interpolation='nearest')
            # # im1=ax1.imshow(image_batch_hr[index].reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            # divider = make_axes_locatable(ax1)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # ax1.axis('off')
            # fig.colorbar(im1, cax=cax, orientation='vertical')

            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image_1[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax3=plt.subplot(dim[0], dim[1], 3)
            imgnr = np.flipud(generated_image_2[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax3.set_title('Recons_f2', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax4=plt.subplot(dim[0], dim[1], 4)
            imgnr = np.flipud(generated_image_3[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax4.set_title('Recons_f3', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
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
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax6.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')


            plt.tight_layout(pad=0.01)
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
            #a=image_batch_hr[index]

def plot_generated_images_clini_all(dirfile,x,y,generator, examples=5, dim=(1, 6), figsize=(10, 5)):
    fg_color = 'black'
    bg_color =  'white'
    PD_label=[]
    GT_label=[]
    global Tmp_ssimlist
    dirfile=dirfile +'/Clinical_predicted_image_'
    label,generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f = generator.predict([x_test_lr_1,x_test_lr_2,x_test_lr_3,x_test_lr_4])
    print (generated_image_1.shape)
    v_min=0.34

    for index in range(x,y):
            fig=plt.figure(figsize=figsize)

            # if (val==False):
            #  image_batch_hr[index]=(image_batch_hr[index]).reshape(128, 128)
            # ax1=plt.subplot(dim[0], dim[1], 1)
            # ax1.set_title('GT', color=fg_color)
            # imgn = np.flipud(image_batch_hr[index]) #/ np.linalg.norm(image_batch_hr[index])
            # im1 = ax1.imshow(imgn.reshape(128, 128))  # , interpolation='nearest')
            # # im1=ax1.imshow(image_batch_hr[index].reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
            # divider = make_axes_locatable(ax1)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # ax1.axis('off')
            # fig.colorbar(im1, cax=cax, orientation='vertical')

            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image_1[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax3=plt.subplot(dim[0], dim[1], 3)
            imgnr = np.flipud(generated_image_2[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax3.set_title('Recons_f2', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ax4=plt.subplot(dim[0], dim[1], 4)
            imgnr = np.flipud(generated_image_3[index]) #/ np.linalg.norm(image_batch_hr[index])
            ax4.set_title('Recons_f3', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
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
            im2=plt.imshow(imgnr.reshape(128, 128),vmin = v_min)#,vmin = _min, vmax = _max)#, interpolation='nearest')
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax6.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')


            plt.tight_layout(pad=0.01)
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
            #a=image_batch_hr[index]
def plot_generated_images(epoch,generator,RTT_model, val =True, examples=5, dim=(1, 6), figsize=(10, 5)):
    import random
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
        dirfile='testcase_mmc/test_generated_image_epoch_'
    # testing timeit()
    starttime = timeit.default_timer()

    print("The start time is :",starttime)

    # label, generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f = RTT_model.predict([image_batch_lr1,image_batch_lr2,image_batch_lr3,image_batch_lr4])
    # label = RTT_model.predict([image_batch_lr1,image_batch_lr2,image_batch_lr3,image_batch_lr4])
    label =  RTT_model.predict([image_batch_lr1,image_batch_lr2,image_batch_lr3,image_batch_lr4])
    print("The time difference is :", (timeit.default_timer() - starttime)/label.shape[0])


    y_pred = np.argmax(label, axis=1)
    # for i in range(3,6):
    #     score=accuracy_score(y_testlabel[i:i+3], y_pred[i:i+3])
    #     print("**********accuracy score ************")
    #     print (score)
    #     i=i+3
    i=12
    score=accuracy_score(y_testlabel[i:i+3], y_pred[i:i+3])
    print("**********accuracy score ************")
    # print (score)
    i=12
    # for i in range(i,i+3):
    #         print (y_testlabel[i], y_pred[i])
#    generated_image = denormalize(gen_img)
#    image_batch_lr = denormalize(image_batch_lr)
    i=12
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    # for index in range(i,i+3):
    #         fig=plt.figure(figsize=figsize)

   
    #         ax1=plt.subplot(dim[0], dim[1], 1)
    #         ax1.set_title('GT', color=fg_color)
    #         imgn = np.flipud(image_batch_hr[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         im1 = ax1.imshow(imgn.reshape(128, 128))
    #         divider = make_axes_locatable(ax1)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax1.axis('off')
    #         fig.colorbar(im1, cax=cax, orientation='vertical')



    #         ax2=plt.subplot(dim[0], dim[1], 2)
    #         imgnr = np.flipud(generated_image_1[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         ax2.set_title('Recons_f1', color=fg_color)
    #         im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
    #         divider = make_axes_locatable(ax2)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax2.axis('off')
    #         fig.colorbar(im2, cax=cax, orientation='vertical')

    #         ax3=plt.subplot(dim[0], dim[1], 3)
    #         imgnr = np.flipud(generated_image_2[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         ax3.set_title('Recons_f2', color=fg_color)
    #         im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
    #         divider = make_axes_locatable(ax3)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax3.axis('off')
    #         fig.colorbar(im2, cax=cax, orientation='vertical')

    #         ax4=plt.subplot(dim[0], dim[1], 4)
    #         imgnr = np.flipud(generated_image_3[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         ax4.set_title('Recons_f3', color=fg_color)
    #         im2=plt.imshow(imgnr.reshape(128, 128))
    #         divider = make_axes_locatable(ax4)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax4.axis('off')
    #         fig.colorbar(im2, cax=cax, orientation='vertical')

    #         ax5=plt.subplot(dim[0], dim[1], 5)
    #         imgnr = np.flipud(generated_image_4[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         ax5.set_title('Recons_f4', color=fg_color)
    #         im2=plt.imshow(imgnr.reshape(128, 128))
    #         divider = make_axes_locatable(ax5)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax5.axis('off')
    #         fig.colorbar(im2, cax=cax, orientation='vertical')

    #         ax6=plt.subplot(dim[0], dim[1], 6)
    #         imgnr = np.flipud(generated_image_f[index]) #/ np.linalg.norm(image_batch_hr[index])
    #         ax6.set_title('Recons_all', color=fg_color)
    #         im2=plt.imshow(imgnr.reshape(128, 128))#,vmin = _min, vmax = _max)#, interpolation='nearest')
    #         divider = make_axes_locatable(ax6)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         ax6.axis('off')
    #         fig.colorbar(im2, cax=cax, orientation='vertical')


    #         plt.tight_layout(pad=0.01)
    #         plt.savefig(dirfile+ '-' +str(index)+'.png' )
    #         v=calculateDistance (generated_image_f[index],image_batch_hr[index])#
    #         DistanceROI.append(v)
    #         p=psnr_torch(generated_image_f[index],image_batch_hr[index])
    #         psnrlist.append(p)
    #         ss_im = ssim(image_batch_hr[index].reshape(128, 128), generated_image_f[index].reshape(128, 128))
    #         ssimlist.append(ss_im)

    #         fjacc= FuzzyJaccard(image_batch_hr[index],generated_image_f[index])
    #         FJaccard.append(fjacc)
    #         plt.close("all")
    #         PD_label.append(label[index])
    #         GT_label.append(y_testlabel[index])
    # FJ_mean= np.mean(FJaccard)
    # FJ_std= np.std(FJaccard)
    # DistanceROI_mean= np.mean(DistanceROI)
    # DistanceROI_std= np.std(DistanceROI)

    # psnr_mean=np.mean(psnrlist)
    # psnr_std=np.std(psnrlist)
    # ssim_mean=np.mean(ssimlist)
    # ssim_std=np.std(ssimlist)

    # loss_file = open('testcase_mmc/losses.txt' , 'a')
    # if (val == True):
    #     loss_file.write('synthe  epoch%d :  DistanceROI = %s + ~  %s ; psnr_mean = %s + ~  %s ; ssim_mean = %s + ~  %s ; FuzzyJaccard_mean = %s + ~ %s \n' %(epoch, DistanceROI_mean, DistanceROI_std,psnr_mean,psnr_std,ssim_mean, ssim_std, FJ_mean, FJ_std ) )
    # if Tmp_ssimlist<ssim_mean:
    #     generator.save('./testcase_mmc/Checkpoint.h5')
    #     print('ssim improved from %s to %s, saving model to weight\n' %(Tmp_ssimlist, ssim_mean))
    #     Tmp_ssimlist = ssim_mean
    # loss_file = open('testcase_mmc/losses_acc.txt' , 'a')
    # if (val == True):
    #     loss_file.write('synthe  epoch%d :  GT_label = %s ; label = %s  \n' %(epoch, GT_label, label ) )

    # loss_file.close()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ##https://scikit-learn.org/0.16/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_testlabel))
    tick_marks = np.arange(3)
    classes=['Healthy', 'Benign','Malignant']
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
def plot_confusionmatrix_singfreq(dir,epoch,RToT_model,x_test,Y_testlabel):

    Y_pred, Im_pred = RToT_model.predict(x_test)
    y_pred = np.argmax(Y_pred, axis=1)
    cm=confusion_matrix(Y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['healthy', 'malign', 'Benign']
    print(classification_report(Y_testlabel, y_pred, target_names=target_names))
    plt.figure()
    plot_confusion_matrix(cm)
    dirfile=dir+'/confusion_matrix_SF'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")
    ##Normalized confusion matrix##
    delta=0.0004
    cm_normalized = np.around(cm.astype('float') /((cm.sum(axis=1)[:, np.newaxis])+delta),decimals=2)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile=dir+'/Normalized_confusion_matrixSF'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")
def plot_confusionmatrix(dirfile,epoch,RToT_model):

    Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    # Y_pred = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    y_pred = np.argmax(Y_pred, axis=1)
    cm=confusion_matrix(y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['Healthy', 'Benign','Malignant']
    print(classification_report(y_testlabel, y_pred, target_names=target_names))
    plt.figure()
    plot_confusion_matrix(cm)
    dirfil=dirfile+'/confusion_matrix'
    plt.savefig(dirfil+ '-'+'.png' )
    plt.close("all")
    y_true =y_testlabel 
    mcc= matthews_corrcoef(y_true, y_pred)
    acc=accuracy_score(y_true, y_pred) 
    bal_acc= balanced_accuracy_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred,average='micro') 
    precision=precision_score(y_true, y_pred,average='weighted') 
    f1=f1_score(y_true, y_pred,average='weighted') 

    print ('mcc_real_RTRT',mcc)
    print ('accuracy_score_real_RTRT',acc)
    print ('balanced_accuracy_score_real_RTRTe',bal_acc)
    print ('recall_score_real_RTRT',recall)
    print ('precision_score_real_RTRT',precision)
    print ('f1_score_real_RTRT',f1)

    # from sklearn import  metrics
    # plt.figure()
    # dirfile='metrics/Roc_curve'
    # metrics.plot_roc_curve(target_names, label_G, y_pred)  # doctest: +SKIP
    # plt.savefig(dirfile+ '-'+'.png' )
    # plt.close("all")

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    delta=0.0004
    cm_normalized = np.around(cm.astype('float') /((cm.sum(axis=1)[:, np.newaxis])+delta),decimals=2)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile=dirfile+'/Normalized_confusion_matrix'

    # dirfile='testcase_mmc/Normalized_confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")
def plot_confusionmatrix_RTT(dirfile,epoch,RToT_model):

    Y_pred = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    # Y_pred = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
    y_pred = np.argmax(Y_pred, axis=1)
    cm=confusion_matrix(y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['Healthy', 'Benign','Malignant']
    print(classification_report(y_testlabel, y_pred, target_names=target_names))
    plt.figure()
    plot_confusion_matrix(cm)
    dirfil=dirfile+'/confusion_matrix_RTT'
    plt.savefig(dirfil+ '-'+'.png' )
    plt.close("all")

    y_pred = np.argmax(Y_pred, axis=1)
    y_true =y_testlabel 
    print ("y_pred", y_pred)
    print ("y_true", y_true)

    mcc= matthews_corrcoef(y_true, y_pred)
    acc=accuracy_score(y_true, y_pred) 
    bal_acc= balanced_accuracy_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred,average='weighted') 
    precision=precision_score(y_true, y_pred,average='weighted') 
    f1=f1_score(y_true, y_pred,average='weighted') 

    print ('mcc_real_RTT',mcc)
    print ('accuracy_score_real_RTT',acc)
    print ('balanced_accuracy_scor_real_RTTe',bal_acc)
    print ('recall_score_real_RTT',recall)
    print ('precision_score_real_RTT',precision)
    print ('f1_score_real_RTT',f1)

    # clf_report = classification_report(y_true,
    #                                y_pred,
    #                                labels=target_names,
    #                                target_names=target_names,
    #                                output_dict=True)
    #                                # .iloc[:-1, :] to exclude support
    # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    # dirfil=dirfile+'/clf_report_RTT'
    # sns.savefig(dirfil+ '-'+'.png' )
    # plt.close("all")

    # from sklearn import  metrics
    # plt.figure()
    # dirfile='metrics/Roc_curve'
    # metrics.plot_roc_curve(target_names, label_G, y_pred)  # doctest: +SKIP
    # plt.savefig(dirfile+ '-'+'.png' )
    # plt.close("all")

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    delta=0.0004
    cm_normalized = np.around(cm.astype('float') /((cm.sum(axis=1)[:, np.newaxis])+delta),decimals=2)
    print('Normalized confusion matrix_RTT')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile=dirfile+'/Normalized_confusion_matrix_RTT'

    # dirfile='testcase_mmc/Normalized_confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

def plot_roc_curve(dirfile,model):
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
    # print (Y_pred)

    lr_probs_ben = Y_pred[:, 1]
    lr_probs_mal = Y_pred[:, 2]
    lr_probs_heal = Y_pred[:, 0]

    # calculate scores
    # lr_auc_ben = roc_auc_score(y[:, 1], lr_probs_ben)
    ns_auc = roc_auc_score(y[:, 0], ns_probs)
    lr_auc_mal = roc_auc_score(y[:, 2], lr_probs_mal)
    lr_auc_H = roc_auc_score(y[:, 0], lr_probs_heal)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    # print('Benign: ROC AUC=%.3f' % (lr_auc_ben))
    print('Malignant: ROC AUC=%.3f' % (lr_auc_mal))
    print('Healthy: ROC AUC=%.3f' % (lr_auc_H))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y[:, 1], ns_probs)
    lr_fprb, lr_tprb, _ = roc_curve(y[:, 1], lr_probs_ben)
    lr_fprm, lr_tprm, _ = roc_curve(y[:, 2], lr_probs_mal)
    lr_fprh, lr_tprh, _ = roc_curve(y[:, 0], lr_probs_heal)


    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='')
    # plt.plot(lr_fprb, lr_tprb, marker='.', label='Benign(ROC AUC=%.3f)' % (lr_auc_ben))
    plt.plot(lr_fprm, lr_tprm, marker='+', label='Malignant(ROC AUC=%.3f)' % (lr_auc_mal))
    plt.plot(lr_fprh, lr_tprh, marker='x', label='Healthy (ROC AUC=%.3f)' % (lr_auc_H))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    dirfile=dirfile+ '/Roc_curve'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

def Compute_ROC_curve_and_area(y_score,y_test):
    from scipy import interp
    from itertools import cycle
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import roc_auc_score
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_score.shape[1]
    print(n_classes) 

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     plt.figure()
    lw = 2
#     plt.plot(fpr[2], tpr[2], color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    classes=['Healthy', 'Benign','Malignant']
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    dirfile='testcase_mmc/Compute_ROC_curve_and_area'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

def  Compute_recall_curve_and_area(y_score,Y_test):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    n_classes = y_score.shape[1]

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    from itertools import cycle
    # setup plot details
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
#         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#         plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

#     lines.append(l)
#     labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"],  color='deeppink', linestyle=':', lw=4)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    classes=['Healthy', 'Benign','Malignant']

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    dirfile='testcase_mmc/Precision-Recall_curve_and_area'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")
    # plt.show()
    
# X_train, y_train, y_label, x_test, y_test, Y_testlabel= a.load_data()
# print("data loaded")
def normalize_test(x,m):
    epsilon=0.01
    mvec = m.mean(0)
    #print(mvec)
    stdvec = m.std(axis=0)
    #print(stdvec)
    return ((x - mvec)/stdvec+epsilon)# s,mvec,stdvec
def plot_MCC(data_1,data_2,data_3,data_4,data_5):
    # Creating dataset
    import matplotlib as mpl

    np.random.seed(18)
    names = ['Raw-to-Task', 'Rec&Diag-Net','FuseNet_L{CE}', 'Cancat_all' 'One_freq']
    data = [data_1, data_2, data_3, data_4,data_5]
    mpl.rcParams["font.size"] = 14

    fig = plt.figure(figsize =(10, 7))
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    # fig.set_fontsize(10)
    # Creating plot
    bp = ax.plot(data, 'ro') #scatter(names,data)
    # show plot
    # plt.show()
    mpl.rcParams["font.size"] = 14

    plt.savefig("testcase_mmc/model_mmc.png", bbox_inches='tight')
    return 0

def plot_MCC_2(data_1,data_2,data_3,data_4,data_5,figname):
    # Creating dataset
    np.random.seed(10)
    
    data = [data_1, data_2, data_3, data_4,data_5]
    
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 1)
    
    colors = ['#0000FF', '#00FF00','#FFFF00', '#FF00FF','#FFF0F0']
    # colors = ['#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF']
    # colors =['#0000FF','#666666','#C0C0C0', '#999999','#000000']# grey scale
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 1)
        
    # x-axis labels
    ax.set_xticklabels(['Raw-to-Task', 'Rec&Diag-Net',
                        'FuseNet_L{CE}', 'Cancat_all','One_freq'])
    
    # Adding title
    # plt.title("Customized box plot")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    # show plot
    # plt.show()
    plt.savefig("testcase_mmc/"+ figname +".png", bbox_inches='tight')
    return 0

######## compute MCC######
measure_1,measure_2,measure_3,measure_4, immatrix, label, testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,immatrix_test ,label_test=a.load_data()

x_test_2= testmeasure_2
x_test_1= testmeasure_1
x_test_3= testmeasure_3
x_test_4= testmeasure_4
# x_train= measure_750



x_test_1 = normalize_data(x_test_1) #here

x_test_2= normalize_data(x_test_2) #here

x_test_3 = normalize_data(x_test_3) #here

x_test_4= normalize_data(x_test_4) #here
# x_test= testmeasure_750
y_testlabel=label_test
y_testima= immatrix_test


x_test_lr_2 = x_test_2
x_test_lr_1 = x_test_1
x_test_lr_4 = x_test_4
x_test_lr_3 = x_test_3


y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  #
x_test_hr=  y_test

print("data loaded")

Y_testlabel=label_binarize(y_testlabel, classes=[0, 1, 2])#
 #np_utils.to_categorical(y_test, 3)
# Y_testlabel=np_utils.to_categorical(y_testlabel, 3)


# #### Load models####

# pathRTT= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_focal_050505_all_05-gam05orth_adjusted fusin_skip0/deep_spa_mse_only.h5'
pathRTT= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_GN persensor_lr/deep_spa_mse_only.h5'
RTT_model= load_model(pathRTT,compile=False)
# Y_pred, Im_pred = recons_model.predict(x_test)
Y_pred = RTT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
# Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RToT_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
y_pred = np.argmax(Y_pred, axis=1)
y_true =y_testlabel 
# print (y_true, y_pred)
m =  100  #80#
mcc_1=[]
bal_acc_1=[]
acc_1=[]
recall_1=[]
precision_1=[]
f1_1=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_1.append(mcc_method_i)
    acc_1.append(acc_method_i)
    bal_acc_1.append(bal_acc_method_i)
    recall_1.append(recall_method_i)
    precision_1.append(precision_method_i)
    f1_1.append(f1_method_i)# print(mcc_1) 
# mcc = matthews_corrcoef(y_true, y_pred)
mcc_1m= matthews_corrcoef(y_true, y_pred)
acc_1m=accuracy_score(y_true, y_pred) 
bal_acc_1m= balanced_accuracy_score(y_true, y_pred)
recall_1m=recall_score(y_true, y_pred,average='micro') 
precision_1m=precision_score(y_true, y_pred,average='micro') 
f1_1m=f1_score(y_true, y_pred,average='macro') 

path= 'nor_orth_diag_02DT_025FJ_skip_0_FUSE_RALL_DIAg_4-6layers_wce/deep_spa_mse_only.h5'
RTRD_model= load_model(path,compile=False)#, custom_objects={'custom_loss_func': loss})
Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RTRD_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
y_pred = np.argmax(Y_pred, axis=1)
mcc_2=[]
bal_acc_2=[]
acc_2=[]
recall_2=[]
precision_2=[]
f1_2=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_2.append(mcc_method_i)
    acc_2.append(acc_method_i)
    bal_acc_2.append(bal_acc_method_i)
    recall_2.append(recall_method_i)
    precision_2.append(precision_method_i)
    f1_2.append(f1_method_i)
mcc_2m= matthews_corrcoef(y_true, y_pred)
acc_2m=accuracy_score(y_true, y_pred) 
bal_acc_2m= balanced_accuracy_score(y_true, y_pred)
recall_2m=recall_score(y_true, y_pred,average='micro') 
precision_2m=precision_score(y_true, y_pred,average='micro') 
f1_2m=f1_score(y_true, y_pred,average='macro') 

path= 'results_fuse_only/deep_spa_mse_only.h5'
RTRD_notorth_model= load_model(path,compile=False)#, custom_objects={'custom_loss_func': loss})
Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RTRD_notorth_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
y_pred = np.argmax(Y_pred, axis=1)
mcc_3=[]
bal_acc_3=[]
acc_3=[]
recall_3=[]
precision_3=[]
f1_3=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_3.append(mcc_method_i)
    acc_3.append(acc_method_i)
    bal_acc_3.append(bal_acc_method_i)
    recall_3.append(recall_method_i)
    precision_3.append(precision_method_i)
    f1_3.append(f1_method_i)
mcc_3m= matthews_corrcoef(y_true, y_pred)
acc_3m=accuracy_score(y_true, y_pred) 
bal_acc_3m= balanced_accuracy_score(y_true, y_pred)
recall_3m=recall_score(y_true, y_pred,average='micro') 
precision_3m=precision_score(y_true, y_pred,average='micro') 
f1_3m=f1_score(y_true, y_pred,average='macro') 

path= 'results_concat_input_CE_orth/deep_spa_mse_only.h5'
# concat all raw to task
path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/AdjustedLoss_Ben/code/results/deep_spa_mse_only.h5'
# concat all join RD to task
path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_JTR_MF_4/New_results_concat_signal_orth_DT_FJ_nofusion/Checkpoint.h5'
RTRD_concat_model= load_model(path,compile=False)#, custom_objects={'custom_loss_func': loss})
Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RTRD_concat_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
# Y_pred = RTRD_concat_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])
y_pred = np.argmax(Y_pred, axis=1)
mcc_4=[]
bal_acc_4=[]
acc_4=[]
recall_4=[]
precision_4=[]
f1_4=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_4.append(mcc_method_i)
    acc_4.append(acc_method_i)
    bal_acc_4.append(bal_acc_method_i)
    recall_4.append(recall_method_i)
    precision_4.append(precision_method_i)
    f1_4.append(f1_method_i)
mcc_4m= matthews_corrcoef(y_true, y_pred)
acc_4m=accuracy_score(y_true, y_pred) 
bal_acc_4m= balanced_accuracy_score(y_true, y_pred)
recall_4m=recall_score(y_true, y_pred,average='micro') 
precision_4m=precision_score(y_true, y_pred,average='micro') 
f1_4m=f1_score(y_true, y_pred,average='macro') 

# path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_JTR/results_drop025_fusion/deep_spa_mse_only.h5'
path= "/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_JTR/rec_dia_onefreq/results/deep_spa_mse_only.h5"
RTRD_freq_model= load_model(path,compile=False)#, custom_objects={'custom_loss_func': loss})
Y_pred, Im_pred_1 = RTRD_freq_model.predict([ x_test_lr_2])
y_pred = np.argmax(Y_pred, axis=1)
mcc_5=[]
bal_acc_5=[]
acc_5=[]
recall_5=[]
precision_5=[]
f1_5=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_5.append(mcc_method_i)
    acc_5.append(acc_method_i)
    bal_acc_5.append(bal_acc_method_i)
    recall_5.append(recall_method_i)
    precision_5.append(precision_method_i)
    f1_5.append(f1_method_i)
mcc_5m= matthews_corrcoef(y_true, y_pred)
acc_5m=accuracy_score(y_true, y_pred) 
bal_acc_5m= balanced_accuracy_score(y_true, y_pred)
recall_5m=recall_score(y_true, y_pred,average='micro') 
precision_5m=precision_score(y_true, y_pred,average='micro') 
f1_5m=f1_score(y_true, y_pred,average='macro') 
pathRTT= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/results_ce_only_adjusted fusin_skip0/deep_spa_mse_only.h5'
RTTCE_model= load_model(pathRTT,compile=False)
# Y_pred, Im_pred = recons_model.predict(x_test)
Y_pred = RTTCE_model.predict([x_test_lr_1, x_test_lr_2, x_test_lr_3,x_test_lr_4])

y_pred = np.argmax(Y_pred, axis=1)
y_true =y_testlabel 
# print (y_true, y_pred)
m =  100  #80#
mcc_6=[]
bal_acc_6=[]
acc_6=[]
recall_6=[]
precision_6=[]
f1_6=[]
for i in range(1,m):
    x_sub, y_sub = zip(*random.sample(list(zip(y_true, y_pred)), m))
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    mcc_method_i = matthews_corrcoef(x_sub, y_sub) 
    acc_method_i=accuracy_score(x_sub, y_sub) 
    bal_acc_method_i=balanced_accuracy_score(x_sub, y_sub) 
    recall_method_i=recall_score(x_sub, y_sub, average='micro') 
    precision_method_i=precision_score(x_sub, y_sub, average='micro') 
    f1_method_i=f1_score(x_sub, y_sub, average='macro') 

    mcc_6.append(mcc_method_i)
    acc_6.append(acc_method_i)
    bal_acc_6.append(bal_acc_method_i)
    recall_6.append(recall_method_i)
    precision_6.append(precision_method_i)
    f1_6.append(f1_method_i)# print(mcc_1) 
# mcc = matthews_corrcoef(y_true, y_pred)
mcc_6m= matthews_corrcoef(y_true, y_pred)
acc_6m=accuracy_score(y_true, y_pred) 
bal_acc_6m= balanced_accuracy_score(y_true, y_pred)
recall_6m=recall_score(y_true, y_pred,average='micro') 
precision_6m=precision_score(y_true, y_pred,average='micro') 
f1_6m=f1_score(y_true, y_pred,average='macro') 

print ('mcc',mcc_1m,mcc_2m,mcc_3m,mcc_4m,mcc_5m,mcc_6m)
# print("here")
print ('accuracy_score',acc_1m,acc_2m,acc_3m,acc_4m,acc_5m,acc_6m)
# print("here")
print ('balanced_accuracy_score',bal_acc_1m,bal_acc_2m,bal_acc_3m,bal_acc_4m,bal_acc_5m,bal_acc_6m)
print ('recall_score',recall_1m,recall_2m,recall_3m,recall_4m,recall_5m,recall_6m)
print ('precision_score',precision_1m,precision_2m,precision_3m,precision_4m,precision_5m,precision_6m)
print ('f1_score',f1_1m,f1_2m,f1_3m,f1_4m,f1_5m,f1_6m)
plot_MCC(mcc_1m,mcc_2m,mcc_3m,mcc_4m,mcc_5m)
plot_MCC_2(mcc_1,mcc_2,mcc_3,mcc_4,mcc_5,"model_mmc_new")
plot_MCC_2(acc_1,acc_2,acc_3,acc_4,acc_5,"acc")
plot_MCC_2(bal_acc_1,bal_acc_2,bal_acc_3,bal_acc_4,bal_acc_5,"bal_acc")
plot_MCC_2(recall_1,recall_2,recall_3,recall_4,recall_5,"recall")
plot_MCC_2(precision_1,precision_2,precision_3,precision_4,precision_5,"precision")
plot_MCC_2(f1_1,f1_2,f1_3,f1_4,f1_5,"f1")

## end compute MCC

# ### compute  Clinical Data Results
testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,label_test=a_clin.load_data(patien_id)
# tr_1,tr_2,tr_3,tr_4, im,trlab,tes1,test2,tes3,tes4,im_test,labtes=a.load_data()
x_sg, y_sg, x_test_i,  y_test_i = a_TL.load_data()

# TLpath='/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/TL/code'
# TLL =load_model(TLpath+'/TL-results_all-128-256/TL.h5',compile=False)#
# print("Loaded TL model from disk")
# TLR =load_model(TLpath+'/TL-results_all-0-128/TL.h5',compile=False)#
# print("Loaded TL model from disk")

# # normalized_clinical_1= normalized_intensity(testmeasure_1)
# # normalized_clinical_2= normalized_intensity(testmeasure_2)
# # normalized_clinical_3= normalized_intensity(testmeasure_3)
# # normalized_clinical_4= normalized_intensity(testmeasure_4)




# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),testmeasure_1[:,0:128]))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),testmeasure_1[:,128:256]))
# Shifted_normalized_clin_Healthy_1 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),testmeasure_2[:,0:128]))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),testmeasure_2[:,128:256]))
# Shifted_normalized_clin_Healthy_2 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),testmeasure_3[:,0:128]))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),testmeasure_3[:,128:256]))
# Shifted_normalized_clin_Healthy_3 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),testmeasure_4[:,0:128]))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),testmeasure_4[:,128:256]))
# Shifted_normalized_clin_Healthy_4 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# x_test_lr_1 = normalize_data(Shifted_normalized_clin_Healthy_1) #here
# x_test_lr_2= normalize_data(Shifted_normalized_clin_Healthy_2) #here
# x_test_lr_3 = normalize_data(Shifted_normalized_clin_Healthy_3) #here
# x_test_lr_4= normalize_data(Shifted_normalized_clin_Healthy_4) #here

# ## uncommnet to remove TL shift ###
# x_test_lr_1 = normalize_data(testmeasure_1) #here
# x_test_lr_2= normalize_data(testmeasure_2) #here
# x_test_lr_3 = normalize_data(testmeasure_3) #here
# x_test_lr_4= normalize_data(testmeasure_4) #here

# # x_test= testmeasure_750
# y_testlabel=label_test
# print (y_testlabel)
# Y_testlabel=label_binarize(y_testlabel, classes=[0, 1, 2])# #np_utils.to_categorical(y_test, 3)


# print("data loaded")
# ## model trained with GN
# path_recon_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/Best_results_GNnoise_mean_Std_fuse_orth/deep_spa_mse_only.h5'
# # path_recon_only= 'results/deep_spa_mse_only.h5'

# R_model= load_model(path_recon_only,compile=False)#, custom_objects={'custom_loss_func': loss})
# # path= 'nor_orth_diag_02DT_025FJ_skip_0_FUSE_RALL_DIAg_4-6layers_wce/deep_spa_mse_only.h5'
# path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/results_GN_randpersorsornoise_m_std_fus_orth/deep_spa_mse_only.h5'
# RTRD_model= load_model(path,compile=False)#, 
# # path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_focal_050505_all_05-gam05orth_adjusted fusin_skip0/deep_spa_mse_only.h5'
# ## noisyRTT model
# path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_GN persensor_lr/deep_spa_mse_only.h5'
# # path_Diag_only= "/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/bestresults_wce_050505_all_03-gam01orth_adjusted fusin_skip0/deep_spa_mse_only.h5"
# D_model= load_model(path_Diag_only,compile=False)#, custom_objects={'custom_loss_func': loss})

# dirfile='Clinical_GNonly-ad/Patient'+str(patien_id)
# plot_generated_images_clini(dirfile, R_model, True)
# plot_confusionmatrix(dirfile,0, R_model)
# plot_confusionmatrix_RTT(dirfile,0, D_model)

# plot_roc_curve(dirfile,RTRD_model)
# # Compute_ROC_curve_and_area(Y_pred,y_testlabel)
# # Compute_recall_curve_and_area(Y_pred,y_testlabel)
# # train(200,64)
# # was 32

## #################  All clinical toghether ###########################
current_directory = "/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/"#os.getcwd()
final_directory = os.path.join(current_directory, 'testcase')
Clinical_directory = os.path.join(current_directory, 'testcase_mmc/Patient_all_rev2')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
if not os.path.exists(Clinical_directory):
   os.makedirs(Clinical_directory)
test_1=[]
test_2=[]
test_3=[]
test_4= []
label_t =[]
one= True
for patien_id in range(1,12):
    print(patien_id)
    if patien_id ==1: # or one:
        testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,label_test=a_clin.load_data(patien_id)
        test_1 = testmeasure_1
        test_2 = testmeasure_2
        test_3 =testmeasure_3
        test_4 = testmeasure_4
        label_t = label_test
        print ("all shape", test_1.shape)
        print ("alllabel  shape", label_t.shape)
    else:    
        if patien_id !=4: #not in [4,5]:#!=4
            testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,label_test=a_clin.load_data(patien_id)
            test_1 = np.concatenate((test_1,testmeasure_1[2:,:]), axis=0)
            test_2 = np.concatenate((test_2,testmeasure_2[2:,:]), axis=0)
            test_3 = np.concatenate((test_3,testmeasure_3[2:,:]), axis=0)
            test_4 = np.concatenate((test_4,testmeasure_4[2:,:]), axis=0)
            label_t = np.concatenate((label_t,label_test[2:]), axis=0)
print ("all shape", test_1.shape)
print ("alllabel  shape", label_t.shape)


## TL module###

TLpath='/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/TL/code'
TLL =load_model(TLpath+'/TL-results_wind_flatness_L/TL.h5', compile=False)#TL-results_flatness_L1/TL.h5 #results_large_L#results_adj2_L#/TL-results_new-0-128/TL.h5
print("Loaded TL model from disk")
TLR =load_model(TLpath+'/TL-results_wind_flatness_R/TL.h5',compile=False)#TL-results_adj2_R #TL-results_new-128-256/TL.h5
print("Loaded TL model from disk")

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),test_1[:,0:128]/test_1.max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),test_1[:,128:256]/test_1.max()))
# Shifted_normalized_clin_Healthy_1 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),test_2[:,0:128]/test_2.max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),test_2[:,128:256]/test_2.max()))
# Shifted_normalized_clin_Healthy_2 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),test_3[:,0:128]/test_3.max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),test_3[:,128:256]/test_3.max()))
# Shifted_normalized_clin_Healthy_3 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg.max(),test_4[:,0:128]/test_4.max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg.max(),test_4[:,128:256]/test_4.max()))
# Shifted_normalized_clin_Healthy_4 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)


# ## norm only 
Shifted_clini_L =TLL.predict(normalize_data(test_1[:,0:128]))
Shifted_clini_R =TLR.predict(normalize_data(test_1[:,128:256]))
Shifted_normalized_clin_Healthy_1 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

Shifted_clini_L =TLL.predict(normalize_data(test_2[:,0:128]))
Shifted_clini_R =TLR.predict(normalize_data(test_2[:,128:256]))
Shifted_normalized_clin_Healthy_2 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

Shifted_clini_L =TLL.predict(normalize_data(test_3[:,0:128]))
Shifted_clini_R =TLR.predict(normalize_data(test_3[:,128:256]))
Shifted_normalized_clin_Healthy_3 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

Shifted_clini_L =TLL.predict(normalize_data(test_4[:,0:128]))
Shifted_clini_R =TLR.predict(normalize_data(test_4[:,128:256]))
Shifted_normalized_clin_Healthy_4 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# ## norm_ref 
# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg[:,0:128].max(),test_1[:,0:128]/test_1[:,0:128].max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg[:,128:256].max(),test_1[:,128:256]/test_1[:,128:256].max()))
# Shifted_normalized_clin_Healthy_1 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg[:,0:128].max(),test_2[:,0:128]/test_2[:,0:128].max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg[:,128:256].max(),test_2[:,128:256]/test_2[:,128:256].max()))
# Shifted_normalized_clin_Healthy_2 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg[:,0:128].max(),test_3[:,0:128]/test_3[:,0:128].max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg[:,128:256].max(),test_3[:,128:256]/test_3[:,128:256].max()))
# Shifted_normalized_clin_Healthy_3 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

# Shifted_clini_L =TLL.predict(normalize_data_ref(x_sg[:,0:128]/x_sg[:,0:128].max(),test_4[:,0:128]/test_4[:,0:128].max()))
# Shifted_clini_R =TLR.predict(normalize_data_ref(x_sg[:,128:256]/x_sg[:,128:256].max(),test_4[:,128:256]/test_4[:,128:256].max()))
# Shifted_normalized_clin_Healthy_4 = np.concatenate((Shifted_clini_L,Shifted_clini_R), axis=1)

test_1 = Shifted_normalized_clin_Healthy_1
test_2= Shifted_normalized_clin_Healthy_2
test_3 = Shifted_normalized_clin_Healthy_3
test_4= Shifted_normalized_clin_Healthy_4
##


## uncommnet to remove TL shift ###
# x_test_lr_1 = normalize_data_ref(tr_1,test_1) #here
# x_test_lr_2= normalize_data_ref(tr_2,test_2) #here
# x_test_lr_3 = normalize_data_ref(tr_3,test_3) #here
# x_test_lr_4= normalize_data_ref(tr_4,test_4) #here

x_test_lr_1 = normalize_data(test_1) #here
x_test_lr_2= normalize_data(test_2) #here
x_test_lr_3 = normalize_data(test_3) #here
x_test_lr_4= normalize_data(test_4) #here

y_testlabel=label_t
Y_testlabel=label_binarize(y_testlabel, classes=[0, 1, 2])# #np_utils.to_categorical(y_test, 3)

# # GN trained model
path_recon_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/Best_results_GNnoise_mean_Std_fuse_orth/deep_spa_mse_only.h5'
# path_recon_only= 'results/deep_spa_mse_only.h5'

R_model= load_model(path_recon_only,compile=False)#, custom_objects=

# path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/results_GN_randpersorsornoise_m_std_fus_orth/deep_spa_mse_only.h5'
# RTRD_model= load_model(path,compile=False)#, 
# path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_GN persensor_lr/deep_spa_mse_only.h5'
# # path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/Best_results_GNnoise_mean_Std_fuse_orth/deep_spa_mse_only.h5'
# # checkpoint_path='/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/Best_results_GNnoise_mean_Std_fuse_orth/weights-improvement-03.hdf5'
# path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/results/deep_spa_mse_only.h5'
# checkpoint_path='/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/results/weights-improvement-02.hdf5'
# D_model= load_model(path_Diag_only,compile=False)#, custom_objects={'custom_loss_func': loss})
# # D_model.load_weights(checkpoint_path)

dirfile=Clinical_directory

## no noise trained model
# path_Diag_only= "/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/best_results_wce_050505_all_05-gam07orth_adjusted fusin_skip0"
path_Diag_only= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_T/bestresults_wce_050505_all_03-gam01orth_adjusted fusin_skip0/'
D_model_path= path_Diag_only+ '/deep_spa_mse_only.h5'

D_model= load_model(D_model_path,compile=False)#
checkpoint_path= path_Diag_only+'/weights-improvement-04.hdf5' #04
D_model.load_weights(checkpoint_path)

# path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/nor_orth_diag_02DT_025FJ_skip_0_FUSE_RALL_DIAg_4-6layers_wce/deep_spa_mse_only.h5'

#path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/results_sensornoise_GN_F_O/deep_spa_mse_only.h5'
path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/results_GN01_fusion_orth/deep_spa_mse_only.h5'
# path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth/results_GN_randpersorsornoise_m_std_fus_orth/deep_spa_mse_only.h5'

RTRD_model= load_model(path,compile=False)#, 

plot_confusionmatrix(dirfile,0, RTRD_model)
plot_confusionmatrix_RTT(dirfile,0, D_model)


#####   Signle frequency #####

path= '/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/R_To_JTR/results_drop025_fusion/deep_spa_mse_only.h5'
SF_model= load_model(path,compile=False)#, custom_objects={'custom_loss_func': loss})
# Y_pred, Im_pred_1 = RTRD_freq_model.predict([ x_test_lr_2])
SF_model= load_model(path,compile=False)#, 



# plot_confusionmatrix(dirfile,0, RTRD_model)
# plot_confusionmatrix_RTT(dirfile,0, D_model)
plot_confusionmatrix_singfreq(dirfile,1,SF_model,x_test_lr_2,y_testlabel)
# for i in range (0,4):
# 	 tes=test_1[i]#.reshape(128, 128)
# 	 plt.plot(tes.T,'b-')
# 	 plt.savefig(dirfile +'/sigg '+ str(i)+'.png', bbox_inches=None)
# 	 plt.close('all')
import shutil
shutil.copy2('/local-scratch/Hanene/DOT_model_2019/new/rnn/MFDL/new_fusion_orth//compute_results.py', Clinical_directory #dirfile
+'/compute_results.py') # complete target filename given






