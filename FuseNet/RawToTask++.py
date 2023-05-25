import argparse
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import label_binarize
from LoadData import load_data, preprocess
from Models import *
from Tools import *
from Utils_models import *
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import itertools
import pandas as pd
from sklearn import preprocessing
from Models import *
from keras.utils import np_utils
from keras import losses
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import array
import skimage
#print(skimage.__version__)
from  skimage.metrics import structural_similarity as ssim
import os
from sklearn.preprocessing import MinMaxScaler
import keras  as keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import timeit
import math
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import ModelCheckpoint

disable_eager_execution()

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')
logger.setLevel(logging.INFO)

from numpy.random import seed



current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'results')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

def initializer(name=None,logs={}):
        global lgr
        configuration = {'epochs':25,'loss':'mse', 'lr':0.00001, 'seed':2, 'device':'gpu', 'arch':'Raw-to-task++', 'batchsize':16, 'alpha':0.2, 'beta':0.25, 'gamma':0.5, 
                  'checkpoint': None, 'datasetdirectory':'/local-scratch/Hanene/Data/multi-freq/Data/', 'outputfolder': "results", 'checkpointdirectory':'.', 'mode':'train'}

        
        parser = argparse.ArgumentParser(description='Optional app description')
        parser.add_argument('--epochs', type=int, nargs='?', help='Int, >0, Epochs, default 25')
        parser.add_argument('--batchsize', type=int, nargs='?', help='Int, >0, batchsize, default 16')
        parser.add_argument('--outputfolder', type=str, nargs='?', help='Output folder')
        parser.add_argument('--mode', type=str, nargs='?', help='train [def], evaluate, resume')
        parser.add_argument('--arch', type=str, nargs='?', help='Raw-to-task++ [def], Raw-to-task')
        parser.add_argument('--datasetdirectory', type=str, nargs='?', help='Path where tif images are stored')
        parser.add_argument('--lr', type=float, nargs='?', help='Float, >0, Learning Rate, default 0.0001')
        # parser.add_argument('--checkpointdirectory', type=str, nargs='?', help='checkpoint directory to resume')
        # parser.add_argument('--checkpoint', type=str, nargs='?', help='checkpoint file to load for evaluation')
        args = parser.parse_args()
        overrides = []
        for k in configuration:
            try:
                argk = getattr(args, k)
                if argk is not None:
                    overrides.append("Overriding {} : {} -> {}".format(k, configuration[k], argk))
                    configuration[k] = argk
            except AttributeError as e:
                continue
        OUTPUTROOT = configuration['outputfolder']
        # outputdirectory = os.path.join(OUTPUTROOT)#, "{}_{}_{}".format(datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss"), str(np.random.randint(1000)), configuration['seed']))
        current_directory = os.getcwd()
        outputdirectory = os.path.join(current_directory, OUTPUTROOT)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory) 
        configuration['outputdirectory'] = outputdirectory
        configuration['logdir'] = outputdirectory
        lgr = initlogger(configuration)
        lgr.info("Writing output in {}".format(outputdirectory))
        lgr.info("Logging directory {}".format(configuration['logdir']))
        lgr.debug("CONF::\t Using configuration :: ")
        for k, v in configuration.items():
            lgr.info("CONF::\t\t {} -> {}".format(k, v))
        # torch.manual_seed(configuration['seed'])
        seed(1)
        tf.random.set_seed(2)
        configuration['logger']=lgr
        return configuration


def train(epochs, batch_size, alpha,beta,gamma,arch,dir):
    alpha = K.variable(alpha)
    beta = K.variable(beta)
    gamma = K.variable(gamma)
    shape = (256,)
    keras.callbacks.Callback()
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=0, mode='auto') #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
    change_lr = LearningRateScheduler(scheduler)
    filepath= dir+'/weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, verbose=1,  monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
    feature_cla, model = Models(shape).RTT_model()
    model_loss_fuse= loss(alpha, beta,batch_size,feature_cla,gamma)
    if arch== 'Raw-to-task++':
 
        model.compile(   
            loss =  { "category_output":  model_loss_fuse },
            metrics = {"reconstruction_output_fuse": 'accuracy' },
            optimizer=tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        )
    elif arch== 'Raw-to-task':
        feature_cla, model = Models(shape).RTT_model()
        model_loss_fuse= loss(alpha, beta,batch_size,feature_cla,gamma)
        model_loss= loss_r(alpha, beta,batch_size) 
        model.compile (   
            loss =  { "category_output": tf.keras.losses.CategoricalCrossentropy() },
            metrics = { "reconstruction_output_fuse": 'accuracy'  },
            optimizer=tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        )


    history =model.fit([measure_1, measure_2, measure_3, measure_4], [label,x_train, x_train, x_train, x_train, x_train], epochs=epochs, batch_size=batch_size, shuffle=True,
    validation_split=0.1, callbacks = [plot_losses,checkpoint,LearningRateReducerCb(),reduce_lr])
    
    plot_confusionmatrix(epochs, model,dir, testmeasure_1, testmeasure_2, testmeasure_3, testmeasure_4, label_test)
    # plot_roc_curve(model)

if __name__ == "__main__":
    conf=initializer()
    arch=conf['arch']
    batchsize= conf['batchsize']  
    lgr=conf['logger']
    alpha =conf['alpha']
    beta = conf['beta']
    gamma = conf['gamma']
    epochs= conf['epochs'] 
    logging.captureWarnings(True)
    print(arch)
    dataset_dir = conf['datasetdirectory']
    outputfolder=  conf['outputfolder']
    print (dataset_dir)
    measure_1,measure_2,measure_3,measure_4, x_train, label, testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,x_test ,label_test=load_data(dataset_dir)
    print("data loaded")
    train(epochs,batchsize, alpha,beta,gamma,arch,outputfolder )

