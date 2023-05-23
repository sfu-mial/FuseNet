import argparse
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import label_binarize
from LoadData import load_data, preprocess
from Models import *
from Tools import getlogger, initlogger
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
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import array
import skimage
print(skimage.__version__)
from  skimage.metrics import structural_similarity as ssim
import os
from sklearn.preprocessing import MinMaxScaler
import keras  as keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import timeit
import math
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')
logger.setLevel(logging.INFO)

from numpy.random import seed


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

def initializer(name=None,logs={}):
        global lgr
        configuration = {'epochs':25,'loss':'mse', 'lr':0.00001, 'seed':2, 'device':'gpu', 'arch':'FuseNet', 'batchsize':16,
                  'checkpoint': None, 'datasetdirectory':'/local-scratch/Hanene/Data/multi-freq/Data/', 'outputfolder': "results", 'checkpointdirectory':'.', 'mode':'train'}

        
        parser = argparse.ArgumentParser(description='Optional app description')
        parser.add_argument('--epochs', type=int, nargs='?', help='Int, >0, Epochs, default 25')
        parser.add_argument('--batchsize', type=int, nargs='?', help='Int, >0, batchsize, default 16')
        parser.add_argument('--outputfolder', type=str, nargs='?', help='Output folder')
        parser.add_argument('--mode', type=str, nargs='?', help='train [def], evaluate, resume')
        parser.add_argument('--arch', type=str, nargs='?', help='FuseNet [def], FuseNet++, Raw-to-task, Raw-to-task++, SF-JRD,SF-DP')
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
        outputdirectory = os.path.join(current_directory, 'outputfolder')
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


def train(epochs=1, batch_size=64):
    alpha = K.variable(0.2)#(0.1)
    beta = K.variable(0.25)#(0.02)
    gamma = K.variable(0.5)#(0.02)
    global Tmp_ssimlist
    Tmp_ssimlist = 0

    shape = (256,)
    feature_cla, model = Models(shape).JRD_model()
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)#,clipvalue=1.0) # was r=1E-5 here
    model_loss_fuse= loss(alpha, beta,batch_size,feature_cla,gamma)#feature_cla
    model_loss= loss_r(alpha, beta,batch_size) 
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

    history =model.fit([measure_1, measure_2, measure_3, measure_4], [label,x_train, x_train, x_train, x_train, x_train], epochs=epochs, batch_size=batch_size, shuffle=True,
    validation_split=0.1,
    # validation_data=(x_test_lr,x_test_hr),
     callbacks = [MyCallback(alpha, beta), plot_losses,checkpoint,LearningRateReducerCb(),reduce_lr])#,lr_decay_callback,change_lr


if __name__ == "__main__":
    conf=initializer()
    arch=conf['arch']
    epochs= conf['epochs'] 
    batchsize= conf['batchsize']  
    lgr=conf['logger']
    logging.captureWarnings(True)
    print(arch)
    dataset_dir = conf['datasetdirectory']
    print (dataset_dir)
    measure_1,measure_2,measure_3,measure_4, x_train, label, testmeasure_1,testmeasure_2,testmeasure_3,testmeasure_4,immatrix_test ,label_test=load_data(dataset_dir)
    print("data loaded")
    train(epochs,batchsize)



# was 32

