import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions


from keras.datasets import fashion_mnist, mnist
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
#from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

##############################################################################
#start #https://www.kaggle.com/code/milan400/fer2013-denoising-using-autoencoder-and-unet
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose


def build_model(input_layer, start_neurons):
    # Input layer
    inputs = (input_layer)  # Assuming RGB images
    start_neurons=64
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)


    # Output layer
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv7) 
    
    
    return decoded


start_neurons=64

input_image=Input(shape=(299, 299, 3))
decoded=build_model(input_image,start_neurons)

model = Model(input_image, decoded)
model.compile(optimizer='adam', loss='MSE')
model.summary()



#####################################################################################################


#make the split of the data


##
#Checkpoint to save the best weights during training
# model_checkpoint = ModelCheckpoint('u_net_weights.h5', save_best_only=True, save_weights_only=True, mode='min', monitor='val_loss')
# epochs = 50  # You can adjust this based on your needs
# model_unet.fit(x_train_noisy, x_train,
#                 epochs=epochs,
#                 batch_size=64,
#                 shuffle=True,
#                 validation_data=(x_test_noisy, x_test),
#                 callbacks=[model_checkpoint])






