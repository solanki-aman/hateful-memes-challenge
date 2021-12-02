import time
import os

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow import keras
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from image_utils import transfer_learning_model, hidden_layers_cnn
from text_utils import remove_punctuations, remove_stopwords, build_word_set
from utils import show_interactive_performance_plot, get_image_arrays

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start_time = time.time()
config = dict()
scores = dict()

image_path = '/home/amansolanki/datasets/hateful-memes-images/'
df = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/train.csv')

EPOCHS = 50
batch_size = 32

# Features and Labels
text = df['text']
image_column = df['image_id']
image_arrays = get_image_arrays(image_column, image_path)
label = df['label']

config['_num_labels'] = label.nunique()
config['_batch_size'] = batch_size

# Train Test Split - Text
X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(text, label, test_size=0.25, random_state=42,
                                                                      shuffle=True)
# Train Test Split - Image
X_train_image, X_val_image, y_train_image, y_val_image = train_test_split(image_arrays, label, test_size=0.25,
                                                                          random_state=42, shuffle=True)

training_set_baseline = y_train_text.value_counts(normalize=True)[0]
validation_set_baseline = y_val_text.value_counts(normalize=True)[0]


scores['training_baseline_accuracy'] = round(training_set_baseline, 4)
scores['validation_baseline_accuracy'] = round(validation_set_baseline, 4)

# ----------------------------------------------------------------------------------------------------------------------
# Text Model Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

# Clean Training Data
temp_train = X_train_text.map(remove_punctuations)
X_train_clean = temp_train.map(remove_stopwords)
# Clean Validation Data
temp_test = X_train_text.map(remove_punctuations)
X_val_clean = temp_test.map(remove_stopwords)

# Build Vocabulary
word_set = build_word_set(X_train_clean.to_list())
num_unique_words = len(word_set)
config['vocab_size'] = num_unique_words

# Prepare Train Data
train_sentences = X_train_text.to_numpy()
train_labels = keras.utils.to_categorical(y_train_text, 2)
# Prepare Validation Data
val_sentences = X_val_text.to_numpy()
val_labels = keras.utils.to_categorical(y_train_text, 2)

# Initialize Text Tokenizer
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(X_train_clean.to_list())

# Transform Sentence Words to Tokens
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# Create Padded Train and Validation Sequences for Model Training
sentence_max_length = max([len(sentence.split()) for sentence in train_sentences])  # 47
max_length = 30
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
config['max_sentence_length'] = sentence_max_length
config['max_length'] = max_length
config["padding"] = "max_length"

# Glove Word Embeddings
# Embedding Transfer Learning
path_to_glove_file = '/home/amansolanki/datasets/glove.42B.300d.txt'

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

num_tokens = num_unique_words + 2
embedding_dim = 300
hits = 0
misses = 0

# Prepare Embedding Matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

config['glove_matched'] = hits
config['glove_missed'] = misses

# ----------------------------------------------------------------------------------------------------------------------
# Image Model Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

# Prepare Train Data
train_labels = keras.utils.to_categorical(y_train_image, 2)
# Prepare Validation Data
val_labels = keras.utils.to_categorical(y_val_image, 2)

# Prepare the training dataset dataloader.
training_batch_generator = DataLoader(image_array=X_train_image, labels=train_labels, batch_size=batch_size)
# Prepare the validation dataset dataloader.
validation_batch_generator = DataLoader(image_array=X_val_image, labels=val_labels, batch_size=batch_size)

# ----------------------------------------------------------------------------------------------------------------------
# Text Model
# ----------------------------------------------------------------------------------------------------------------------
lstm_model = keras.Input(shape=(max_length,), name='lstm_model_input')
embedding = keras.layers.Embedding(input_dim=num_tokens,
                                   output_dim=embedding_dim,
                                   input_length=max_length,
                                   embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                   trainable=False)(lstm_model)
# model.add(layers.Embedding(input_dim=num_unique_words, output_dim=512, input_length=max_length))
lstm1 = keras.layers.LSTM(8, return_sequences=True, kernel_regularizer='l2')(embedding)
lstm2 = keras.layers.LSTM(8, dropout=0.1, kernel_regularizer='l2')(lstm1)
flatten = keras.layers.Flatten()(lstm2)
dense1 = keras.layers.Dense(16, activation='relu', kernel_regularizer='l2')(flatten)
dropout1 = keras.layers.Dropout(0.2)(dense1)
dense3 = keras.layers.Dense(6, activation='relu', kernel_regularizer='l2')(dropout1)
flatten = keras.layers.Flatten()(dense3)
lstm_layer = keras.layers.Dense(6, activation='relu', kernel_regularizer='l2')(flatten)

# ----------------------------------------------------------------------------------------------------------------------
# Image Model
# ----------------------------------------------------------------------------------------------------------------------
image_model = keras.Input(shape=(224, 224, 3), name='image_model_input')
# hidden_layers = hidden_layers_cnn(image_model)
hidden_layers = hidden_layers_cnn(image_model)
batch_normalization_1 = keras.layers.BatchNormalization()(hidden_layers)
dense1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(batch_normalization_1)
batch_normalization_2 = keras.layers.BatchNormalization()(dense1)
dense2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(batch_normalization_2)
flatten = keras.layers.Flatten()(dense2)
image_layer = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(flatten)

# ----------------------------------------------------------------------------------------------------------------------
# Multimodal Model
# ----------------------------------------------------------------------------------------------------------------------
merged_input = keras.layers.concatenate([lstm_layer, image_layer])
merged_dense1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(merged_input)
merged_batch_normalization_1 = keras.layers.BatchNormalization()(merged_dense1)
merged_dense2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(merged_batch_normalization_1)
merged_dropout = keras.layers.Dropout(0.6)(merged_dense2)
merged_dense3 = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(merged_dropout)
model_output = keras.layers.Dense(2, activation='sigmoid', name='label_output')(merged_dense3)
model = keras.models.Model(inputs=[image_model, lstm_model], outputs=[model_output])

model.summary()

keras.utils.plot_model(model, to_file='/home/amansolanki/PycharmProjects/hateful-memes-challenge/plots'
                                      '/multimodal_image_text_architecture.png', dpi=300)

loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=1e-08)
# optimizer = keras.optimizers.Adadelta(learning_rate=1e-04, rho=0.95, epsilon=1e-07)

losses = {
    'label_output': loss
}

metrics = {
    'label_output': 'accuracy'
}

model.compile(
    optimizer=optimizer,
    loss=losses,
    metrics=metrics
)

train_X_multi_input = {
    'image_model_input': X_train_image,
    'lstm_model_input': train_padded
}

multimodal_model = model.fit(train_X_multi_input,
                             train_labels,
                             epochs=EPOCHS,
                             batch_size=batch_size,
                             verbose=1,
                             validation_split=0.1)

show_interactive_performance_plot(multimodal_model, 'multimodal_model_accuracy', 'accuracy', 'val_accuracy')
show_interactive_performance_plot(multimodal_model, 'multimodal_model_loss', 'loss', 'val_loss')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('multimodal_model.tflite', 'wb') as f:
    f.write(tflite_model)

val_X_multi_input = {
    'image_model_input': X_val_image,
    'lstm_model_input': val_padded
}

loss, training_accuracy = model.evaluate(train_X_multi_input, train_labels, verbose=1)
scores['training_accuracy'] = round(training_accuracy, 4)

loss, validation_accuracy = model.evaluate(val_X_multi_input, val_labels, verbose=1)
scores['validation_accuracy'] = round(validation_accuracy, 4)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 1 Predictions')
test_seen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_seen = test_seen_original.copy()

test_seen['text'] = test_seen.text.map(remove_punctuations)
test_seen['text'] = test_seen.text.map(remove_stopwords)
test_seen_image_column = test_seen['image_id']

test_seen_baseline = test_seen['label'].value_counts(normalize=True)[0]
scores['test_seen_baseline_accuracy'] = round(test_seen_baseline, 4)

test_sentences = test_seen.text.to_numpy()
test_seen_labels = keras.utils.to_categorical(test_seen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Prepare Data for Evaluation
test_seen_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")
test_seen_image_arrays = get_image_arrays(test_seen_image_column, image_path)

test_seen_multi_input = {
    'image_model_input': test_seen_image_arrays,
    'lstm_model_input': test_seen_padded
}

loss, test_seen_accuracy = model.evaluate(test_seen_multi_input, test_seen_labels, verbose=1)
scores['test_seen_accuracy'] = round(test_seen_accuracy, 4)

preds = model.predict(test_seen_multi_input)
df_seen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_seen.to_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/image-text-concat/predictions'
               '/test_seen_multimodal.csv', index=False)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 2 Predictions')
test_unseen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')
test_unseen = test_unseen_original.copy()

test_unseen['text'] = test_unseen.text.map(remove_punctuations)
test_unseen['text'] = test_unseen.text.map(remove_stopwords)
test_unseen_image_column = test_unseen['image_id']

test_unseen_baseline = test_unseen['label'].value_counts(normalize=True)[0]
scores['test_unseen_baseline_accuracy'] = round(test_seen_baseline, 4)

test_sentences = test_unseen.text.to_numpy()
test_unseen_labels = keras.utils.to_categorical(test_unseen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Prepare Data for Evaluation
test_unseen_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")
test_unseen_image_arrays = get_image_arrays(test_unseen_image_column, image_path)

test_unseen_multi_input = {
    'image_model_input': test_unseen_image_arrays,
    'lstm_model_input': test_unseen_padded
}

loss, test_unseen_accuracy = model.evaluate(test_unseen_multi_input, test_unseen_labels, verbose=1)
scores['test_unseen_accuracy'] = round(test_unseen_accuracy, 4)

preds = model.predict(test_unseen_multi_input)
df_unseen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_unseen.to_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/image-text-concat/predictions'
                 '/test_unseen_multimodal.csv', index=False)

with open('config.json', 'w') as fp:
    json.dump(config, fp, indent=4)

with open('scores.json', 'w') as fp:
    json.dump(scores, fp, indent=4)
