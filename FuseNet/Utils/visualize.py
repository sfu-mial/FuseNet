
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from Utils.Utils_models import *
from sklearn.preprocessing import StandardScaler
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json
import os
import itertools
import numpy as np
import pandas as pd
import glob
import math
from numpy import genfromtxt
from keras import optimizers
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import keras as keras
from math import sqrt
from sklearn.utils import shuffle
from sklearn import manifold
from keras import backend as K
from sklearn.metrics import mean_squared_error,median_absolute_error
from keras.models import load_model
import timeit
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
from  skimage.metrics import structural_similarity as ssim
import skimage
from keras import losses
from sklearn.preprocessing import label_binarize

global Tmp_ssimlist
Tmp_ssimlist = 0

def plot_generated_images(epoch, dir,generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f, x_train, GT_label,val =True, examples=20, dim=(1, 6), figsize=(10, 5)):
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
    # PD_label=[]
    # GT_label=[]
    global Tmp_ssimlist
    dirfile= dir+ '/test_generated_image_'
    if val :
        r= examples
    else: 
        r= len(x_train)
    for index in range(r):
            ## plot GT
            fig=plt.figure(figsize=figsize)
            ax1=plt.subplot(dim[0], dim[1], 1)
            ax1.set_title('GT', color=fg_color)
            imgn = np.flipud(x_train[index]) 
            im1 = ax1.imshow(imgn.reshape(128, 128))  
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.axis('off')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            ## plot Rec1
            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image_1[index]) 
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec2
            ax3=plt.subplot(dim[0], dim[1], 3)
            imgnr = np.flipud(generated_image_2[index]) 
            ax3.set_title('Recons_f2', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec3
            ax4=plt.subplot(dim[0], dim[1], 4)
            imgnr = np.flipud(generated_image_3[index]) 
            ax4.set_title('Recons_f3', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax4.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec4
            ax5=plt.subplot(dim[0], dim[1], 5)
            imgnr = np.flipud(generated_image_4[index]) 
            ax5.set_title('Recons_f4', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax5.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec_fusion
            ax6=plt.subplot(dim[0], dim[1], 6)
            imgnr = np.flipud(generated_image_f[index]) 
            ax6.set_title('Recons_all', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax6.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.tight_layout(pad=0.01)
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
                
            ## compute metrics
            v=calculateDistance (generated_image_f[index],x_train[index])#
            DistanceROI.append(v)
            p=psnr(generated_image_f[index],x_train[index])
            psnrlist.append(p)
            ss_im = ssim(x_train[index].reshape(128, 128), generated_image_f[index].reshape(128, 128))
            ssimlist.append(ss_im)
            fjacc= FuzzyJaccard(x_train[index],generated_image_f[index])
            FJaccard.append(fjacc)
            plt.close("all")
 
    FJ_mean= np.mean(FJaccard)
    FJ_std= np.std(FJaccard)
    DistanceROI_mean= np.mean(DistanceROI)
    DistanceROI_std= np.std(DistanceROI)
    psnr_mean=np.mean(psnrlist)
    psnr_std=np.std(psnrlist)
    ssim_mean=np.mean(ssimlist)
    ssim_std=np.std(ssimlist)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ##https://scikit-learn.org/0.16/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
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

def plot_confusionmatrix(epoch,dir, y_pred,y_testlabel):
    cm=confusion_matrix(y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['healthy', 'Benign', 'malign']
    print(classification_report(y_testlabel, y_pred, target_names=target_names))


    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],decimals=2)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile=dir+'/Normalized_confusion_matrix'
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

def plot_Metrics(data_1,data_2,data_3,data_4,data_5,figname):
    # Creating dataset
    np.random.seed(10)
    
    data = [data_1, data_2, data_3, data_4,data_5]
    
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 1)
    
    colors = ['#0000FF', '#00FF00','#FFFF00', '#FF00FF','#FFF0F0']

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
    plt.savefig("testcase_mmc/"+ figname +".png", bbox_inches='tight')
    return 0


