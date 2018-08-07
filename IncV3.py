# coding: utf-8
# # InceptionV3 on MURA
# Directly using InceptionV3 from Keras: https://keras.io/applications
import argparse
from datetime import datetime
from os import environ

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
#local package
import utils

#import numpy as np
#from matplotlib import pyplot as plt

#from keras.applications.resnet50 import ResNet50
#from keras.applications import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import load_model
from keras import optimizers
#use multi-gpu & fine-tune model
#from keras.utils.training_utils import multi_gpu_model
#use new file name after install the BN fix version Keras 2.1.6 fork
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.models import Sequential

# args section
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--gpus', default=2, type=int)
parser.add_argument('--classes', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
parser.add_argument('--lr-wait', default=10, type=int, help='how long to wait on plateu')
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--fullretrain', dest='fullretrain', action='store_true', help='retrain all layers of the model')
parser.add_argument('--seed', default=1337, type=int, help='random seed')
parser.add_argument('--img_channels', default=3, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--early_stop', default=20, type=int)
parser.add_argument('--tune_layer', default=249, type=int)

# define global/local variables
fine_tune = True
Loss_Use = 'binary_crossentropy' #categorical_crossentropy

global args
args = parser.parse_args()
# args section end
now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

#import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) #walter: auto detect running kernel
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
#import CLR
from clr_callback import CyclicLR
clr_triangular = CyclicLR(base_lr=0.001, max_lr=0.006, mode='triangular', step_size=256) #step size = 2*batch_size


#############################################################################################
#define model
#############################################################################################
# To convert images to 224*224*3
# To apply random lateral inversions and rotations.
#########################

# get the training data from pickle files
x_train, y_train, w_train = utils.read_pickle2array(sample='train', y_output=3)

# We then scale the variable-sized images to 224x224
# We augment .. by applying random lateral inversions and rotations.
train_datagen = ImageDataGenerator(
rescale=1. / 255,
rotation_range=45,
# width_shift_range=0.2,
# height_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

# fit all training data, be careful of the size
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=False, sample_weight=w_train)

# get the validation data from pickle files
x_valid, y_valid = utils.read_pickle2array(sample='valid', y_output=2)

val_datagen = ImageDataGenerator(rescale=1. / 255)
# fit all training data, be careful of the size
val_datagen.fit(x_valid)
val_generator = val_datagen.flow(x_valid, y_valid, batch_size=args.batch_size, shuffle=True)

#number of samples
n_of_train_samples = x_train.shape[0]
n_of_val_samples = x_valid.shape[0]

# Architectures
img_shape = (args.img_size, args.img_size, args.img_channels)

if args.gpus >= 2:
    #need to allocate model to CPU first to coordinate the model process
    with tf.device('/device:CPU:0'):
        # create the base pre-trained model
        base_model = InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet')
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer, turn-off to try
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        # n_classes; softmax for multi-class, sigmoid for binary
        #predictions = Dense(14, activation='softmax')(x)
        predictions = Dense(args.classes, activation='sigmoid', name='predictions')(x)
        # this is the model we will train
        bmodel = Model(inputs=base_model.input, outputs=predictions)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
else:
    # create the base pre-trained model
    base_model = InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet')
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer, turn-off to try
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    # n_classes; softmax for multi-class, sigmoid for binary
    #predictions = Dense(14, activation='softmax')(x)
    predictions = Dense(args.classes, activation='sigmoid', name='predictions')(x)
    # this is the model we will train
    bmodel = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
"""
base_model = MobileNet(input_shape=img_shape, weights='imagenet', include_top=False)
x = base_model.output  # Recast classification layer
# x = Flatten()(x)  # Uncomment for Resnet based models
x = GlobalAveragePooling2D(name='predictions_avg_pool')(x)  # comment for RESNET models
# n_classes; softmax for multi-class, sigmoid for binary
x = Dense(args.classes, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
"""
# checkpoints
#
checkpoint = ModelCheckpoint(filepath='./models/InceptionV3.hdf5', verbose=1, save_best_only=True)
early_stop = EarlyStopping(patience=args.early_stop)
tensorboard = TensorBoard(log_dir='./logs/InceptionV3/{}/'.format(now_iso))
# reduce_lr = ReduceLROnPlateau(factor=0.03, cooldown=0, patience=args.lr_wait, min_lr=0.1e-6)
callbacks = [checkpoint, clr_triangular]

# Calculate class weights: comment out if sample weight is in use
#weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
#weights = {0: weights[0], 1: weights[1]}

# for layer in base_model.layers:
#     layer.set_trainable = False

# print(model.summary())
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)
if args.resume:
    bmodel.load_weights(args.resume)
    #for layer in bmodel.layers:
    #    layer.set_trainable = True

# set fine tune layers
if fine_tune == True:
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in bmodel.layers[:args.tune_layer]:
       layer.trainable = False
    for layer in bmodel.layers[args.tune_layer:]:
       layer.trainable = True
    print('Unfreeze top N inception blocks')

#use multi gpus
from modelMGPU import ModelMGPU
# check parameter
if args.gpus >= 2:
    #use multi gpus
    model = ModelMGPU(bmodel, gpus=args.gpus)
else:
    model = bmodel


# The network is trained end-to-end using Adam with default parameters
model.compile(
    #optimizer=Adam(lr=args.lr, decay=args.decay),
    #optimizer=SGD(lr=args.lr, decay=args.decay,momentum=args.momentum, nesterov=True),
    optimizer=RMSprop(lr=args.lr),
    loss=f'{Loss_Use}',
    metrics=[binary_accuracy], )

model_out = model.fit_generator(
    train_generator,
    steps_per_epoch=n_of_train_samples // args.batch_size,
    epochs=args.epochs,
    validation_data=val_generator,
    validation_steps=n_of_val_samples // args.batch_size,
    #class_weight=weights,
    workers=args.workers,
    use_multiprocessing=True,
    callbacks=callbacks)

