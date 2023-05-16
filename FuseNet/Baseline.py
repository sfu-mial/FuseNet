# Single Frequency reconstruction models usded as baseline 
from Utils_models import *
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import keras  as keras
from keras.layers import *
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from keras import backend as K
from keras.utils.vis_utils import plot_model

class Models(object):

    def __init__(self, input_shape):
        
        self.input_shape = input_shape
    ##  SingleFreq_JRD_model
    def SF_JRD_model(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

        ''' Add noise '''
	    input0 = keras.Input(shape = self.input_shape, name="vec1")
	    input0 = GaussianNoise(0.1)(input0)

        '''build_reconstruction_branchs'''
		# final image #1
	    model1 = reconst_block(input0, 32, initializers=normal, shape= target_shape)
	    out_r1 = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same", name="reconstruction_output1")(model1)
	   
        '''Task_branch'''
	    feature_clas,  out_r_task = diagnosis_block(out_r1, 64, 3, 1)

	    generator_model = Model(inputs = [input0],  outputs = [out_r_task,out_r1])
	    generator_model.summary([])
	    return feature_clas, generator_model

    ##  SingleFreq_DP_model
     def SF_DP_model(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

        ''' Add noise '''
	    input0 = keras.Input(shape = self.input_shape, name="vec1")
	    input0 = GaussianNoise(0.1)(input0)

        '''Task_branch'''
	    feature_clas,  out_r_task = diagnosis_block(input0, 64, 3, 1)

	    generator_model = Model(inputs = [input0],  outputs = [out_r_task])
	    generator_model.summary([])
	    return feature_clas, generator_model

