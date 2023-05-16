
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

    def JRD_model(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

        ''' Add noise '''
	    input0 = keras.Input(shape = self.input_shape, name="vec1")
	    input0_n = GaussianNoise(0.1)(input0)

	    input1 = keras.Input(shape = self.input_shape, name="vec2")
	    input1_n = GaussianNoise(0.1)(input1)#0.01

	    input2 = keras.Input(shape = self.input_shape, name="vec3")
	    input2_n = GaussianNoise(0.1)(input2)
        
	    input3 = keras.Input(shape = self.input_shape, name="vec4")
	    input3_n = GaussianNoise(0.1)(input3)

        ''' Fuse #freqs '''
	    gen_input =Fusion(input0_n, input1_n, input2_n, input3_n)

        '''build_reconstruction_branchs'''
		# final image #1
	    model1 = reconst_block(input0, 32, initializers=normal, shape= target_shape)
	    out_r1 = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same", name="reconstruction_output1")(model1)

	    # final image #2
	    model2 = reconst_block(input1, 32, initializers=normal, shape= target_shape)
	    out_r2 = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same", name="reconstruction_output2")(model2)

		# final image #3
	    model3 = reconst_block(input2, 32, initializers=normal, shape= target_shape)
	    out_r3 = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same",name="reconstruction_output3")(model3)

		# final image #4
	    model4 = reconst_block(input3, 32, initializers=normal, shape= target_shape)
	    out_r4= Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same", name="reconstruction_output4")(model4)

        #reconstruction_fusion
        model5 = reconst_block(gen_input, 32, initializers=normal, shape= target_shape)

	    # model4= Dense( 128*128, activation = 'relu', kernel_initializer='glorot_normal')(gen_input)
	    # model4 = keras.layers.Reshape(target_shape)(model4)
	    # for index in range(4): #6
	    #     model4 = res_block_gen(model4, 3, 32, 1)

	    out_r_fuse = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same",name="reconstruction_output_fuse",kernel_regularizer=tf.keras.regularizers.L2(0.001))(model5)


        '''Task_branch'''
	
	    feature_clas,  out_r_task = diagnosis_block(out_r_fuse, 64, 3, 1)

	    generator_model = Model(inputs = [input0, input1, input2, input3],  outputs = [out_r_task,out_r1, out_r2, out_r3, out_r4,out_r_fuse])
	    generator_model.summary([])
	    return feature_clas, generator_model

        def RTT_model(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

        ''' Add noise '''
	    input0 = keras.Input(shape = self.input_shape, name="vec1")
	    input0_n = GaussianNoise(0.1)(input0)

	    input1 = keras.Input(shape = self.input_shape, name="vec2")
	    input1_n = GaussianNoise(0.1)(input1)#0.01

	    input2 = keras.Input(shape = self.input_shape, name="vec3")
	    input2_n = GaussianNoise(0.1)(input2)
        
	    input3 = keras.Input(shape = self.input_shape, name="vec4")
	    input3_n = GaussianNoise(0.1)(input3)

        ''' Fuse #freqs '''
	    gen_input =Fusion(input0_n, input1_n, input2_n, input3_n)

       
        '''Task_branch'''
	
	    feature_clas,  out_r_task = diagnosis_block(gen_input, 64, 3, 1)

	    generator_model = Model(inputs = [input0, input1, input2, input3],  outputs = [out_r_task,out_r1, out_r2, out_r3, out_r4,out_r_fuse])
	    generator_model.summary([])
	    return feature_clas, generator_model

