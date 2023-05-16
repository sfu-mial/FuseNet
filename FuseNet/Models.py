
# Modules
from keras.layers import Dense
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization
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
# from tensorflow.keras.models import *
# Residual block
def Fusion(vec1, vec2, vec3,vec4):
    skip= 1,
    dim=32
    ratio= 0.02
    use_bilinear=1, 
    gate1=1,
    gate2=1, 
    gate3=1, 
    gate4=1, 
    dim1=32,
    dim2=32, 
    dim3=32, 
    scale_dim1=1,
    scale_dim2=1, 
    scale_dim3=1,
    mmhid=96, 
    dropout_rate=0.25
    
    # skip_dim = dim1+dim2+dim3+3 if skip else 0

    ### Gated Multimodal Units
    if gate1:
        h1 = Dense(dim, activation = 'relu')(vec1)
        print("gate attention vec1-vec2")

        fuse = tf.keras.layers.Concatenate(axis=1)([vec2, vec3,vec4])
        z1 = Dense(dim)(fuse) # Gate Path with Omic
        z1 = Activation('sigmoid')(z1) 
        multi= z1* h1# tf.keras.layers.Multiply()([z1, h1])
        
        o1 = Dense( dim, activation = 'relu')(multi)
        o1= Dropout(ratio)(o1)
    else:
        print("Noooo gate attention vec1-vec2")

        o1 = Dense( dim, activation = 'relu')(vec1)
        o1= Dropout(ratio)(o1)

    if gate2:
        h2 = Dense(dim, activation = 'relu')(vec2)

        fuse2 = tf.keras.layers.Concatenate(axis=1)([vec1, vec3,vec4])
        z2 = Dense(dim)(fuse2) # Gate Path with Omic
        z2 = Activation('sigmoid')(z2) 

        multi2= z2* h2#tf.keras.layers.Multiply()([z2, h2])
        o2 = Dense( dim, activation = 'relu')(multi2)
        o2= Dropout(ratio)(o2)
    else:
        o2 = Dense( dim, activation = 'relu')(vec2)
        o2 = Dropout(ratio)(o2)

    if gate3:
        h3 = Dense(dim, activation = 'relu')(vec3)

        fuse3 = tf.keras.layers.Concatenate(axis=1)([vec1, vec2, vec4])
        z3 = Dense(dim)(fuse3) # Gate Path with Omic
        z3 = Activation('sigmoid')(z3) 
        multi3=  z3*h3#tf.keras.layers.Multiply()([z3, h3])
        o3 = Dense( dim, activation = 'relu')(multi3)
        o3= Dropout(ratio)(o3)
    else:
        o3 = Dense( dim, activation = 'relu')(vec3)
        o3= Dropout(ratio)(o3)
    
    if gate4:
        h4 = Dense(dim, activation = 'relu')(vec4)

        fuse4 = tf.keras.layers.Concatenate(axis=1)([vec1, vec2, vec3])
        z4 = Dense(dim)(fuse4) #
        z4 = Activation('sigmoid')(z4) 
        multi4=z4*h4#tf.keras.layers.Multiply()([z4, h4])
        o4 = Dense( dim, activation = 'relu')(multi4)
        o4= Dropout(ratio)(o4)
    else:
        o4 = Dense( dim, activation = 'relu')(vec4)
        o4= Dropout(ratio)(o4)

    ### Fusion
 
    print("dists {.shape}".format(o1))
    print("dists {.shape}".format(o2))
    print("dists {.shape}".format(o3))

    one_tens1= K.tf.ones ((K.tf.shape(o1)[0],1),dtype='float32')#(128,128)
    one_tens2= K.tf.ones ((K.tf.shape(o2)[0],1),dtype='float32')#(128,128)
    one_tens3= K.tf.ones ((K.tf.shape(o3)[0],1),dtype='float32')#(128,128)
    print("dists_ones {.shape}".format(one_tens1))

    o1= Concatenate(axis=1)([o1,  one_tens1])
    print("dists_concat {.shape}".format(o1))

    o2= Concatenate(axis=1)([o2, one_tens2])
    o3= Concatenate(axis=1)([o3, one_tens3])
    o4= Concatenate(axis=1)([o4,  one_tens1])
    concat=  Concatenate(axis=1)([o1, o2, o3, o4])
    # o1 =tf.expand_dims(o1, 1)
    # print("distso1 {.shape}".format(o1))

    o1 =tf.expand_dims(o1, 2)

    o2 =tf.expand_dims(o2, 1)
    print("distso2{.shape}".format(o2))

    # o12= tf.keras.layers.Dot(axes=(2, 1))([o1, o2])
    o12= tf.keras.layers.Multiply()([o1, o2])

    print("distso12{.shape}".format(o12))

    o12= tf.reshape(o12,[K.tf.shape(o1)[0],(dim+1)*(dim+1)])
    print("distso12{.shape}".format(o12))

    o12 =tf.expand_dims(o12, 2)
    print("distso12{.shape}".format(o12))
    o3 =tf.expand_dims(o3, 1)
    print("distso3{.shape}".format(o3))

    # o123 = tf.keras.layers.Dot(axes=(2, 1))([o12, o3])
    o123 = tf.keras.layers.Multiply()([o12, o3])

    print("distsoo123{.shape}".format(o123))

    o123= tf.reshape(o123,[K.tf.shape(o1)[0],(dim+1)*(dim+1)*(dim+1)])
    print("distsoo123{.shape}".format(o123))

    o123 =tf.expand_dims(o123, 2)
    print("distso123{.shape}".format(o123))
    o4 =tf.expand_dims(o4, 1)
    print("dists_o4 {.shape}".format(o4))
    # o1234 = tf.keras.layers.Dot(axes=(2, 1))([o123, o4])
    o1234 = tf.keras.layers.Multiply()([o123, o4])

    print("distsoo1234{.shape}".format(o1234))

    o1234= tf.reshape(o1234,[K.tf.shape(o1)[0],(dim+1)*(dim+1)*(dim+1)*(dim+1)])
    print("distsoo1234{.shape}".format(o1234))

    out_fused = Dropout(0.5)(o1234)
    out_fused = Dense(128, activation = 'relu')(out_fused)
    out_fused =  Dropout(ratio)(out_fused)
    if skip:
         print("gated attention_concat_only no kronecker fusion")
         out = concat
    else:
         out = out_fused
         print("********gated attention with kroneckerfusion*********")

    out = Dense(128, activation = 'relu')(out)
    out =  Dropout(ratio, name="fusion")(out)

    return concat, out      
def res_block_gen(model, kernal_size, filters, strides):

    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    # model = BatchNormalization()(model)
    model = Activation('tanh')(model)

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    # model = BatchNormalization()(model)
    model = Activation('tanh')(model)
 
#
   # squeeze and excite spacial wise block
    model = squeeze_excite_spacial_wise_block(model) # x2

##     Concurrent Spatial and Channel Squeeze & Excitation
#    model = add([x1, x2])

    model = add([gen, model])

    return model
    
def squeeze_excite_spacial_wise_block(input):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = init._keras_shape[channel_axis]
    se = Conv2D(filters = 1, kernel_size = 1, strides = 1, padding = "same")(init)
    se = Activation('sigmoid')(se) 

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def squeeze_excite_channel_wise_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='glorot_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='glorot_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
    
def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

def reconst_block(input, filters, initializers, shape):
    # create an image
    model= Dense( 128*128, activation = 'relu', kernel_initializer=initializers)(input)
    model = keras.layers.Reshape(shape)(model)
    # convolve over image
    model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('relu')(model) 
    # attention over image
    for index in range(4): #16
        model = res_block_gen(model, 3, 32, 1)
    # for index in range(1): #16
	#      model2= res_block_gen(model2, 5, 32, 1)
#	    model = res_block_gen(model, 5, 32, 1)
    # regularise  image
    model= Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same",kernel_regularizer=tf.keras.regularizers.L2(0.001))(model)
    model = Activation('relu')(model)
    return model

class Models(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def JRD(self):
	    target_shape = [128, 128,1]
	    normal=keras.initializers.he_normal(seed=None)

	    input0 = keras.Input(shape = self.noise_shape, name="vec1")
	    input0_n = GaussianNoise(0.1)(input0)

	    input1 = keras.Input(shape = self.noise_shape, name="vec2")
	    input1_n = GaussianNoise(0.1)(input1)#0.01

	    input2 = keras.Input(shape = self.noise_shape, name="vec3")
	    input2_n = GaussianNoise(0.1)(input2)
        
	    input3 = keras.Input(shape = self.noise_shape, name="vec4")
	    input3_n = GaussianNoise(0.1)(input3)

	    # gen_input = tf.keras.layers.Concatenate()([input0, input1, input2, input3])
	    feature, gen_input =Fusion(input0_n, input1_n, input2_n, input3_n)
	    # gen_input = Activation("relu", name="21")(gen_input)

#	    x = GaussianNoise(0.01)(gen_input)

        ##build_reconstruction_branch1
		# final image #1
        # x = GaussianNoise(0.02)(gen_input)
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

        ##build_reconstruction_fuse_branch

	    # fuse_recons = tf.keras.layers.Concatenate()([out_r1, out_r2, out_r3, out_r4])
	    # model3 = Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same",kernel_regularizer=tf.keras.regularizers.L1L2(0.01))(fuse_recons)
        # x = model3

	    # model_task= Dense( 128*128, activation = 'relu', kernel_initializer='glorot_normal')(gen_input)
	    # model_task = keras.layers.Reshape(target_shape)(model_task)
	    # model_task = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model_task)
	    # model_task = Activation('relu')(model_task) 
	    # x = model_task

	    # for index in range(1): #16
	    #     x = res_block_gen(x, 3, 32, 1)

	    # model3= x
	    model4= Dense( 128*128, activation = 'relu', kernel_initializer='glorot_normal')(gen_input)
	    # model4 = tf.keras.layers.Concatenate()([model4, out_r1, out_r2, out_r3,out_r4])

	    model4 = keras.layers.Reshape(target_shape)(model4)
	    for index in range(6): #16
	        model4 = res_block_gen(model4, 3, 32, 1)

	    # model4 = reconst_block(model4, 32, initializers=normal, shape= target_shape)

	    out_r_fuse = Conv2D(filters = 1, kernel_size = 7, strides = 1,kernel_initializer='glorot_normal', padding = "same",name="reconstruction_output_fuse",kernel_regularizer=tf.keras.regularizers.L2(0.001))(model4)


        ##build_task_branch
	    # model_task= Dense( 128*128, activation = 'relu', kernel_initializer='glorot_normal')(gen_input)
	    # model_task = keras.layers.Reshape(target_shape)(model_task)
	    # model_task = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model_task)
	    # model_task = Activation('relu')(model_task) 
	    # model_task = MaxPooling2D(pool_size=(2, 2))(model_task)
	    model_task = MaxPooling2D(pool_size=(2, 2))(out_r_fuse)

	    model_task = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model_task)
	    model_task = Activation('relu')(model_task) 

	    model_task = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model_task)
	    model_task = Activation('relu')(model_task) 
	    model_task = MaxPooling2D(pool_size=(2, 2))(model_task)

	    model_task = Flatten()(model_task)
	    model_task = Dense(32)(model_task)
	    model_task = Activation("relu")(model_task)
	    feature_clas= model_task
	    model_task = Dense(3)(model_task)
	    out_r_task = Activation('softmax', name="category_output")(model_task)

	    generator_model = Model(inputs = [input0, input1, input2, input3],  outputs = [out_r_task,out_r1, out_r2, out_r3, out_r4,out_r_fuse])
	    generator_model.summary([])
	    # tf.keras.utils.plot_model(model, to_file='results/model.png', show_shapes=False, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
	    # keras.utils.plot_model(generator_model)
	    return gen_input,feature_clas, generator_model

