import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, concatenate, Activation
from tensorflow.keras.optimizers import Adam
K.set_image_data_format("channels_last")
# Set the image shape to have the channels in the last dimension

def Unet():
    '''
    Builds the 3D UNet Keras model.
    Depth of UNet model = 4.
    returns:
        Untrained 3D UNet Model.
    '''

    input_layer = tf.keras.Input(shape=(16, 160, 160, 4))
    
    down_depth_0_layer_0 = Conv3D(filters=32, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(input_layer)
    down_depth_0_layer_0 = Activation('relu')(down_depth_0_layer_0)
    
    down_depth_0_layer_1 = Conv3D(filters=64, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_0_layer_0)
    down_depth_0_layer_1 = Activation('relu')(down_depth_0_layer_1)
    
    down_depth_0_layer_pool = MaxPooling3D(pool_size=(2,2,2))(down_depth_0_layer_1)
    
    down_depth_1_layer_0 = Conv3D(filters=64, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_0_layer_pool)
    down_depth_1_layer_0 = Activation('relu')(down_depth_1_layer_0)
    
    down_depth_1_layer_1 = Conv3D(filters=128, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_1_layer_0)
    down_depth_1_layer_1 = Activation('relu')(down_depth_1_layer_1)
    
    down_depth_1_layer_pool = MaxPooling3D(pool_size=(2,2,2))(down_depth_1_layer_1)
    
    down_depth_2_layer_0 = Conv3D(filters=128, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_1_layer_pool)
    down_depth_2_layer_0 = Activation('relu')(down_depth_2_layer_0)
   
    down_depth_2_layer_1 = Conv3D(filters=256, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_2_layer_0)
    down_depth_2_layer_1 = Activation('relu')(down_depth_2_layer_1)
    
    down_depth_2_layer_pool = MaxPooling3D(pool_size=(2,2,2))(down_depth_2_layer_1)
    
    down_depth_3_layer_0 = Conv3D(filters=256, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_2_layer_pool)
    down_depth_3_layer_0 = Activation('relu')(down_depth_3_layer_0)
    
    down_depth_3_layer_1 = Conv3D(filters=512, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(down_depth_3_layer_0)
    down_depth_3_layer_1 = Activation('relu')(down_depth_3_layer_1)
    
    up_depth_2_layer_0 = UpSampling3D(size=(2,2,2))(down_depth_3_layer_1)
    
    up_depth_2_concat = concatenate([up_depth_2_layer_0,down_depth_2_layer_1],axis=4)
    
    up_depth_2_layer_1 = Conv3D(filters=256, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_2_concat)
    up_depth_2_layer_1 = Activation('relu')(up_depth_2_layer_1)
    
    up_depth_2_layer_2 = Conv3D(filters=256, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_2_layer_1)
    up_depth_2_layer_2 = Activation('relu')(up_depth_2_layer_2)
   
    up_depth_1_layer_0 = UpSampling3D(size=(2,2,2))(up_depth_2_layer_2)
    
    up_depth_1_concat = concatenate([up_depth_1_layer_0,down_depth_1_layer_1],axis=4)
  
    up_depth_1_layer_1 = Conv3D(filters=128, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_1_concat)
    up_depth_1_layer_1 = Activation('relu')(up_depth_1_layer_1)
    
    up_depth_1_layer_2 = Conv3D(filters=128, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_1_layer_1)
    up_depth_1_layer_2 = Activation('relu')(up_depth_1_layer_2)
    
    up_depth_0_layer_0 = UpSampling3D(size=(2,2,2))(up_depth_1_layer_2)
    
    up_depth_0_concat = concatenate([up_depth_0_layer_0,down_depth_0_layer_1],axis=4)
    
    up_depth_0_layer_1 = Conv3D(filters=64, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_0_concat)
    up_depth_0_layer_1 = Activation('relu')(up_depth_0_layer_1)
    
    up_depth_0_layer_2 = Conv3D(filters=64, kernel_size=(3,3,3),padding='same',strides=(1,1,1))(up_depth_0_layer_1)
    up_depth_0_layer_2 = Activation('relu')(up_depth_0_layer_2)
    
    final_conv = Conv3D(filters=3, kernel_size=(1,1,1),padding='valid',strides=(1,1,1))(up_depth_0_layer_2)
    final_activation = Activation('sigmoid')(final_conv)
    
    model = Model(inputs=input_layer, outputs=final_activation)
    model.compile(optimizer=Adam(learning_rate=0.00001),loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    # model.summary()

    return model

def load_model():
    ''' 
    Loads weights from a pre-trained model.
    returns:
        Trained 3D UNet Model.
    '''
    model = Unet()
    model.load_weights('model_pretrained.hdf5')
    #model.summary()
    return model




