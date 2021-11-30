import tensorflow as tf
from tensorflow import keras


def transfer_learning_model(pretrained_model, input_layer):
    base_model = pretrained_model(input_tensor=input_layer, weights='imagenet')
    base_model.trainable = False
    x = base_model.layers[-2].output
    dense1 = keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    dropout = keras.layers.Dropout(0.8)(dense1)
    dense2 = keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(dropout)
    return dense2
