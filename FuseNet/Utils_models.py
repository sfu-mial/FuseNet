
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
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
print(skimage.__version__)
from keras import losses
from sklearn.preprocessing import label_binarize

global Tmp_ssimlist
Tmp_ssimlist = 0
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

def normalize_data(values):
    from math import sqrt

    scaler = StandardScaler()
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)

    return normalized


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

    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],decimals=2)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    dirfile='mse/Normalized_confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")

## losses
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

def OrthogonalProjectionLoss(features,labels, batch_size, gamma=0.5):
    """
    A Keras version of the orthogonal projection loss defined in
    https://arxiv.org/pdf/2103.14021.pdf
    https://github.com/kahnchana/opl
    
    """
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


def dst_transform(y_true, y_pred):  
    """
    Compute the distance transform of y_true, y_pred 
    Then calculate the error between the distance  (L1 or L2) 
    Usage:
    loss = dst_transform(weights)
    """

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
    
        L1_Distance= K.mean(K.sum(K.abs(result_true - result_pred), axis=-1), axis=-1)/(128)      #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))

        loss += L1_Distance
    return loss/ size
def focal_loss(gamma=5., alpha=.25): # binary only
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

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
        
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

def lr_decay(epoch):
        return 0.1 * math.pow(0.1, epoch)

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

def loss(alpha, Beta,batch_size,feature, gamma):
    def custom_loss_func(y_true, y_pred):
        return custom_loss(y_true, y_pred, alpha, Beta,batch_size,feature,gamma)
    return custom_loss_func


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

def plot_generated_images(epoch, generator,dir,measure_1, measure_2, measure_3, measure_4, x_train, GT_label,val =True, examples=20, dim=(1, 6), figsize=(10, 5)):
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
    if (val ==True):
        image_batch_hr = x_train[:,:]
        image_batch_lr1 = measure_1[:,:]
        image_batch_lr2 = measure_2[:,:]
        image_batch_lr3 = measure_3[:,:]
        image_batch_lr4 = measure_4[:,:]
        dirfile= dir+ '/test_generated_image_'
    
    label, generated_image_1 ,generated_image_2, generated_image_3 ,generated_image_4, generated_image_f = generator.predict([image_batch_lr1,image_batch_lr2,image_batch_lr3,image_batch_lr4])
    for index in range(examples):
            if (val==False):
              image_batch_hr[index]=(image_batch_hr[index]).reshape(128, 128)
            ## plot GT
            fig=plt.figure(figsize=figsize)
            ax1=plt.subplot(dim[0], dim[1], 1)
            ax1.set_title('GT', color=fg_color)
            imgn = np.flipud(image_batch_hr[index]) 
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
            v=calculateDistance (generated_image_f[index],image_batch_hr[index])#
            DistanceROI.append(v)
            p=psnr(generated_image_f[index],image_batch_hr[index])
            psnrlist.append(p)
            ss_im = ssim(image_batch_hr[index].reshape(128, 128), generated_image_f[index].reshape(128, 128))
            ssimlist.append(ss_im)
            fjacc= FuzzyJaccard(image_batch_hr[index],generated_image_f[index])
            FJaccard.append(fjacc)
            plt.close("all")
            # PD_label.append(label[index])
            # GT_label.append(y_testlabel[index])
    FJ_mean= np.mean(FJaccard)
    FJ_std= np.std(FJaccard)
    DistanceROI_mean= np.mean(DistanceROI)
    DistanceROI_std= np.std(DistanceROI)
    psnr_mean=np.mean(psnrlist)
    psnr_std=np.std(psnrlist)
    ssim_mean=np.mean(ssimlist)
    ssim_std=np.std(ssimlist)
    loss_file = open( dir+ '/losses.txt' , 'a')
    if (val == True):
        loss_file.write('synthe  epoch%d :  DistanceROI = %s + ~  %s ; psnr_mean = %s + ~  %s ; ssim_mean = %s + ~  %s ; FuzzyJaccard_mean = %s + ~ %s \n' %(epoch, DistanceROI_mean, DistanceROI_std,psnr_mean,psnr_std,ssim_mean, ssim_std, FJ_mean, FJ_std ) )
    if Tmp_ssimlist<ssim_mean:
        generator.save( dir+ '/Checkpoint.h5')
        print('ssim improved from %s to %s, saving model to weight\n' %(Tmp_ssimlist, ssim_mean))
        Tmp_ssimlist = ssim_mean
    loss_file = open( dir+ '/losses_acc.txt' , 'a')
    if (val == True):
        loss_file.write('synthe  epoch%d :  GT_label = %s ; label = %s  \n' %(epoch, GT_label, label ) )
    loss_file.close()

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

def plot_confusionmatrix(epoch,RToT_model,dir, measure_1, measure_2, measure_3, measure_4, y_label):
    Y_pred, Im_pred_1,Im_pred_2, Im_pred_3,Im_pred_4, Im_pred_f = RToT_model.predict([measure_1, measure_2, measure_3, measure_4])
    y_pred = np.argmax(Y_pred, axis=1)

    y_testlabel= np.argmax(y_label,1) 
    print(y_testlabel)
    print(y_pred)

    cm=confusion_matrix(y_testlabel, y_pred)
    print('Classification Report')
    target_names = ['healthy', 'Benign', 'malign']
    print(y_testlabel.shape, y_pred.shape)
    print(classification_report(y_testlabel, y_pred, target_names=target_names))
    Classification_Report_file = open(dir+'/losses.txt' , 'a')

    plt.figure()
    plot_confusion_matrix(cm)
    dirfile=dir+'/confusion_matrix'
    plt.savefig(dirfile+ '-'+'.png' )
    plt.close("all")


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



