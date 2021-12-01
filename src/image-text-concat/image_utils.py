import tensorflow as tf
from tensorflow import keras


def transfer_learning_model(pretrained_model, input_layer):
    base_model = pretrained_model(input_tensor=input_layer, include_top=False)
    base_model.trainable = False
    x = base_model.layers[-2].output
    dense1 = keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    dropout = keras.layers.Dropout(0.8)(dense1)
    dense2 = keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(dropout)
    return dense2


def hidden_layers_cnn(input_layer):
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=keras.regularizers.l2(l=0.1))(input_layer)
    maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    batchNorm1 = keras.layers.BatchNormalization()(maxPooling1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=keras.regularizers.l2(l=0.1))(batchNorm1)
    maxPooling2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
    batchNorm2 = keras.layers.BatchNormalization()(maxPooling2)
    conv3 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=keras.regularizers.l2(l=0.1))(batchNorm2)
    maxPooling3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
    batchNorm3 = keras.layers.BatchNormalization()(maxPooling3)
    conv4 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=keras.regularizers.l2(l=0.1))(batchNorm3)
    maxPooling4 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)
    batchNorm4 = keras.layers.BatchNormalization()(maxPooling4)
    flatten = keras.layers.Flatten()(batchNorm4)
    dense1 = keras.layers.Dense(512, activation='relu')(flatten)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(256, activation='relu')(dropout1)
    batchNorm5 = keras.layers.BatchNormalization()(dense2)
    dropout2 = keras.layers.Dropout(0.5)(batchNorm5)
    dense3 = keras.layers.Dense(128, activation='relu')(dropout2)
    return dense3
