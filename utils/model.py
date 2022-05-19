from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dropout, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D,BatchNormalization
from tensorflow import keras
from utils.Config import Config
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input, Dropout, concatenate, Bidirectional


from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential




def get_model(config):
    inputs = Input((config.wave_len, config.channels))
    kernel_size = config.kernel_size
    conv_channels = config.conv_channels
    nClasses = 4
    seed = config.seed
    conv1 = Conv1D(conv_channels, kernel_size, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(conv_channels, kernel_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(conv_channels*2, kernel_size, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(conv_channels*2, kernel_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    conv3 = Conv1D(conv_channels*4, kernel_size, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(conv_channels*4, kernel_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(conv_channels*8, kernel_size, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(conv_channels*8, kernel_size, activation='relu', padding='same')(conv4)

    up1 = Conv1D(conv_channels*4, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv4))
    merge1 = concatenate([up1, conv3], axis=-1)
    conv5 = Conv1D(conv_channels*4, kernel_size, activation='relu', padding='same')(merge1)
    conv5 = Conv1D(conv_channels*4, kernel_size, activation='relu', padding='same')(conv5)
    
    up2 = Conv1D(conv_channels*2, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv5))
    merge2 = concatenate([up2, conv2], axis=-1)
    conv6 = Conv1D(conv_channels*2, kernel_size, activation='relu', padding='same')(merge2)
    conv6 = Conv1D(conv_channels*2, kernel_size, activation='relu', padding='same')(conv6)
    
    up3 = Conv1D(conv_channels, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv6))
    merge3 = concatenate([up3, conv1], axis=-1)
    conv7 = Conv1D(conv_channels, kernel_size, activation='relu', padding='same')(merge3)
    conv7 = Conv1D(conv_channels, kernel_size, activation='relu', padding='same')(conv7)
    
    conv8 = Conv1D(nClasses, 1)(conv7)
    conv9 = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=conv9)

    opt = keras.optimizers.Adam(lr=1e-3)    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    