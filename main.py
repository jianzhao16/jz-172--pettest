# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from keras.optimizers import Adam, SGD
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle

myPathRoot='C:/Users/jzhao/Downloads/data/pet/'

def image_generator(files, batch_size=32, sz=(256, 256)):
    while True:

        # extract a random batch
        batch = np.random.choice(files, size=batch_size)

        # variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        for f in batch:

            # get the masks. Note that masks are png files
            mask = Image.open(f'{myPathRoot}annotations/trimaps/{f[:-4]}.png')
            #mask = Image.open(f'annotations/trimaps/{f[:-4]}.png')
            mask = np.array(mask.resize(sz))

            # preprocess the mask
            mask[mask >= 2] = 0
            mask[mask != 0] = 1

            batch_y.append(mask)

            # preprocess the raw images
            raw = Image.open(f'{myPathRoot}images/{f}')
            raw = raw.resize(sz)
            raw = np.array(raw)

            # check the number of channels because some of the images are RGBA or GRAY
            if len(raw.shape) == 2:
                raw = np.stack((raw,) * 3, axis=-1)

            else:
                raw = raw[:, :, 0:3]

            batch_x.append(raw)

        # preprocess a batch of images and masks
        batch_x = np.array(batch_x) / 255.
        batch_y = np.array(batch_y)
        batch_y = np.expand_dims(batch_y, 3)

        yield (batch_x, batch_y)


batch_size = 32
#all_files = os.listdir('images')
all_files = os.listdir(myPathRoot+'images')
shuffle(all_files)

split = int(0.95 * len(all_files))

#split into training and testing
train_files = all_files[0:split]
test_files = all_files[split:]

train_generator = image_generator(train_files, batch_size=batch_size)
test_generator = image_generator(test_files, batch_size=batch_size)
x, y = next(train_generator)

plt.axis('off')
img = x[0]
msk = y[0].squeeze()
msk = np.stack((msk,)*3, axis=-1)

plt.imshow( np.concatenate([img, msk, img*msk], axis = 1))

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou


def unet(sz=(256, 256, 3)):
    x = Input(sz)
    inputs = x

    # down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64

    # bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1

    # upsampling
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1

    # classification
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[mean_iou])

    return model

def build_callbacks():
    checkpointer = ModelCheckpoint(filepath='unet.h5', verbose=0, save_best_only=True, save_weights_only=True)
    callbacks = [checkpointer, PlotLearning()]
    return callbacks


# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        print('i=', self.i, 'loss=', logs.get('loss'), 'val_loss=', logs.get('val_loss'), 'mean_iou=',
              logs.get('mean_iou'), 'val_mean_iou=', logs.get('val_mean_iou'))

        # choose a random test image and preprocess
        path = np.random.choice(test_files)
        raw = Image.open(f'{myPathRoot}images/{path}')
        raw = np.array(raw.resize((256, 256))) / 255.
        raw = raw[:, :, 0:3]

        # predict the mask
        pred = mymodel.predict(np.expand_dims(raw, 0))

        # mask post-processing
        msk = pred.squeeze()
        msk = np.stack((msk,) * 3, axis=-1)
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        # show the mask and the segmented image
        combined = np.concatenate([raw, msk, raw * msk], axis=1)
        plt.axis('off')
        plt.imshow(combined)
        plt.show()

train_steps = len(train_files) //batch_size
test_steps = len(test_files) //batch_size
#model.fit_generator(train_generator,epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps, callbacks = build_callbacks(), verbose = 0)
mymodel=unet(sz=(256, 256, 3))
#mymodel.fit(train_generator,epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps, callbacks = build_callbacks(), verbose = 0)
mymodel.fit_generator(train_generator,epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps, callbacks = build_callbacks(), verbose = 0)

raw = Image.open('test.jpg')
raw = np.array(raw.resize((256, 256)))/255.
raw = raw[:,:,0:3]

#predict the mask
pred = mymodel.predict(np.expand_dims(raw, 0))

#mask post-processing
msk  = pred.squeeze()
msk = np.stack((msk,)*3, axis=-1)
msk[msk >= 0.5] = 1
msk[msk < 0.5] = 0

#show the mask and the segmented image
combined = np.concatenate([raw, msk, raw* msk], axis = 1)
plt.axis('off')
plt.imshow(combined)
plt.show()