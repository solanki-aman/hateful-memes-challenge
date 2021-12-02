import time
import pandas as pd
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from data_loader import DataLoader
from image_utils import transfer_learning_model, hidden_layers_cnn
from utils import get_image_arrays, show_interactive_performance_plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

start_time = time.time()
config = dict()
scores = dict()

image_path = '/home/amansolanki/datasets/hateful-memes-images/'
df = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/train.csv')

# Features and Labels
image_column = df['image_id']
image_arrays = get_image_arrays(image_column, image_path)
label = df['label']
batch_size = 32
EPOCHS = 50

config['_num_labels'] = label.nunique()
config['_batch_size'] = batch_size

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(image_arrays, label, test_size=0.25, random_state=42, shuffle=True)

training_set_baseline = y_train.value_counts(normalize=True)[0]
scores['training_baseline'] = round(training_set_baseline, 4)
validation_set_baseline = y_val.value_counts(normalize=True)[0]
scores['validation_baseline'] = round(validation_set_baseline, 4)

# Prepare Train Data
train_labels = keras.utils.to_categorical(y_train, 2)
# Prepare Validation Data
val_labels = keras.utils.to_categorical(y_val, 2)

# Prepare the training dataset dataloader.
training_batch_generator = DataLoader(image_array=X_train, labels=train_labels, batch_size=batch_size)
# Prepare the validation dataset dataloader.
validation_batch_generator = DataLoader(image_array=X_val, labels=val_labels, batch_size=batch_size)

# Model
input_layer = keras.Input(shape=(224, 224, 3))
hidden_layers = hidden_layers_cnn(input_layer)
# hidden_layers = transfer_learning_model(keras.applications.ResNet152V2, input_layer)
batch_normalization_1 = keras.layers.BatchNormalization()(hidden_layers)
dense1 = keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(batch_normalization_1)
batch_normalization_2 = keras.layers.BatchNormalization()(dense1)
dense2 = keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(batch_normalization_2)
label_branch = keras.layers.Dense(2, activation='sigmoid', name='label_output')(dense2)
model = keras.models.Model(inputs=input_layer, outputs=[label_branch])
model.summary()

config['_model'] = 'Custom'

optimizer = keras.optimizers.Adam(learning_rate=1e-05)
loss = keras.losses.BinaryCrossentropy()
EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics='accuracy'
)

config['train_datashape'] = X_train.shape
config['test_datashape'] = X_val.shape

gpu = tf.config.experimental.list_physical_devices('GPU')
print(gpu)
print('\n')

image_unimodel = model.fit(X_train, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.1)

'''
image_unimodel = model.fit(training_batch_generator,
                           steps_per_epoch=int(X_train.shape[0] // batch_size),
                           epochs=EPOCHS,
                           verbose=1,
                           validation_data=validation_batch_generator,
                           validation_steps=int(X_val.shape[0] // batch_size),
                           callbacks=[EarlyStoppingCallback]
                           )
'''

show_interactive_performance_plot(image_unimodel, 'image_unimodal_accuracy', 'accuracy', 'val_accuracy')
show_interactive_performance_plot(image_unimodel, 'image_unimodal_loss', 'loss', 'val_loss')

model_save_directory = '/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/model'

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('image_model.tflite', 'wb') as f:
    f.write(tflite_model)

end_time = time.time()
time_taken = end_time - start_time
time_taken_minutes = round(time_taken / 60, 3)

config['epochs'] = EPOCHS
config['training_time'] = time_taken_minutes

loss, training_set_accuracy = model.evaluate(X_train, train_labels, verbose=1)
scores['training_set_accuracy'] = round(training_set_accuracy, 4)
loss, validation_set_accuracy = model.evaluate(X_val, val_labels, verbose=1)
scores['validation_set_accuracy'] = round(validation_set_accuracy, 4)

with open('config.json', 'w') as fp:
    json.dump(config, fp, indent=4)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 1 Predictions')
test_seen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_seen = test_seen_original.copy()

# Features and Labels
test_seen_image_column = test_seen['image_id']
test_seen_label = test_seen['label']
test_seen_baseline = test_seen_label.value_counts(normalize=True)[0]
scores['test_seen_baseline'] = round(test_seen_baseline, 4)

# Prepare Data for Evaluation
test_seen_image_arrays = get_image_arrays(test_seen_image_column, image_path)
labels_seen = keras.utils.to_categorical(test_seen_label, 2)

# Prepare Phase 1 dataset dataloader.
# test_seen_batch_generator = DataLoader(image_array=test_seen_image_arrays, labels=labels_seen, batch_size=batch_size)

loss, test_seen_accuracy = model.evaluate(test_seen_image_arrays, labels_seen, verbose=1)
scores['test_seen_accuracy'] = round(test_seen_accuracy, 4)

preds = model.predict(test_seen_image_arrays)
df_seen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_seen.to_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/predictions/test_seen_image.csv',
               index=False)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 2 Predictions')
test_unseen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')
test_unseen = test_unseen_original.copy()

# Features and Labels
test_unseen_image_column = test_unseen['image_id']
test_unseen_label = test_unseen['label']
test_unseen_baseline = test_unseen_label.value_counts(normalize=True)[0]
scores['test_unseen_baseline'] = round(test_unseen_baseline, 4)

# Prepare Data for Evaluation
test_unseen_image_arrays = get_image_arrays(test_unseen_image_column, image_path)
labels_unseen = keras.utils.to_categorical(test_unseen_label, 2)

# Prepare Phase 2 dataset dataloader.

loss, test_unseen_accuracy = model.evaluate(test_unseen_image_arrays, labels_unseen, verbose=1)
scores['test_unseen_accuracy'] = round(test_unseen_accuracy, 4)

preds = model.predict(test_unseen_image_arrays)
df_unseen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_unseen.to_csv(
    '/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/predictions/test_unseen_image.csv',
    index=False)

with open('scores.json', 'w') as fp:
    json.dump(scores, fp, indent=4)
