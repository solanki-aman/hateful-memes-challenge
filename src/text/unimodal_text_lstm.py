import time
import os
import pickle
import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from text_utils import remove_punctuations, remove_stopwords, build_word_set
from utils import show_interactive_performance_plot


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start_time = time.time()
config = dict()
scores = dict()
df = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/train.csv')

EPOCHS = 200

# Features and Labels
text = df['text']
label = df['label']

config['_num_labels'] = label.nunique()

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(text, label, test_size=0.1, random_state=42, shuffle=True)
training_set_baseline = y_train.value_counts(normalize=True)[0]
validation_set_baseline = y_val.value_counts(normalize=True)[0]
scores['validation_baseline'] = round(validation_set_baseline, 4)
scores['training_baseline'] = round(training_set_baseline, 4)

# Clean Training Data
temp_train = X_train.map(remove_punctuations)
X_train_clean = temp_train.map(remove_stopwords)
# Clean Validation Data
temp_test = X_val.map(remove_punctuations)
X_val_clean = temp_test.map(remove_stopwords)

# Build Vocabulary
word_set = build_word_set(X_train.to_list())
num_unique_words = len(word_set)
config['vocab_size'] = num_unique_words

# Prepare Train Data
train_sentences = X_train_clean.to_numpy()
train_labels = keras.utils.to_categorical(y_train, 2)
# Prepare Validation Data
val_sentences = X_val_clean.to_numpy()
val_labels = keras.utils.to_categorical(y_val, 2)

# Initialize Text Tokenizer
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(X_train.to_list())

# Transform Sentence Words to Tokens
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# Create Padded Train and Validation Sequences for Model Training
sentence_max_length = max([len(sentence.split()) for sentence in train_sentences])  # 47
max_length = 30
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
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


# Model
model = keras.models.Sequential()

model.add(layers.Embedding(input_dim=num_tokens,
                           output_dim=embedding_dim,
                           input_length=max_length,
                           embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                           trainable=False))

# model.add(layers.Embedding(input_dim=num_unique_words, output_dim=512, input_length=max_length))
model.add(layers.LSTM(8, return_sequences=True, kernel_regularizer='l2'))
model.add(layers.LSTM(8, dropout=0.1, kernel_regularizer='l2'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer='l2'))
model.add(layers.Dropout(0.2))
# model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer='l2'))
# model.add(layers.Dropout(0.9))
model.add(keras.layers.Dense(6, activation='relu', kernel_regularizer='l2' ))
model.add(layers.Dense(2, activation="sigmoid"))

model.summary()

loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=1e-08)
# optimizer = keras.optimizers.SGD(learning_rate=1e-09)
# optimizer = keras.optimizers.Adadelta(learning_rate=1e-07, rho=0.80, epsilon=1e-07)
EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)

model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')

text_unimodal = model.fit(train_padded,
                          train_labels,
                          epochs=EPOCHS,
                          validation_split=0.3,
                          batch_size=64,
                          verbose=1)

show_interactive_performance_plot(text_unimodal, 'text_unimodal_accuracy', 'accuracy', 'val_accuracy')
show_interactive_performance_plot(text_unimodal, 'text_unimodal_loss', 'loss', 'val_loss')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('text_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save Tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

end_time = time.time()
time_taken = end_time - start_time
time_taken_minutes = round(time_taken/60, 3)

loss, train_set_accuracy = model.evaluate(train_padded, train_labels, verbose=1)
scores['train_set_accuracy'] = round(train_set_accuracy, 4)

loss, validation_set_accuracy = model.evaluate(val_padded, val_labels, verbose=1)
scores['validation_set_accuracy'] = round(validation_set_accuracy, 4)

config['training_time'] = time_taken_minutes

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('image_model.tflite', 'wb') as f:
    f.write(tflite_model)

with open('config.json', 'w') as fp:
    json.dump(config, fp, indent=4)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 1 Predictions')
test_seen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_seen = test_seen_original.copy()

test_seen['text'] = test_seen.text.map(remove_punctuations)
test_seen['text'] = test_seen.text.map(remove_stopwords)

test_seen_labels = test_seen['label']
test_seen_baseline = test_seen_labels.value_counts(normalize=True)[0]
scores['test_seen_baseline'] = round(test_seen_baseline, 4)

test_sentences = test_seen.text.to_numpy()
test_labels = keras.utils.to_categorical(test_seen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

loss, test_seen_accuracy = model.evaluate(test_padded, test_labels, verbose=1)
print('Phase 1 accuracy: {:5.2f}%'.format(100 * test_seen_accuracy))
scores['test_seen_accuracy'] = round(test_seen_accuracy, 4)

preds = model.predict(test_padded)
df_seen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_seen.to_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_seen_text.csv', index=False)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 2 Predictions')
test_unseen_original = pd.read_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')
test_unseen = test_unseen_original.copy()

test_unseen['text'] = test_unseen.text.map(remove_punctuations)
test_unseen['text'] = test_unseen.text.map(remove_stopwords)

test_unseen_label = test_unseen['label']
test_unseen_baseline = test_unseen_label.value_counts(normalize=True)[0]
scores['test_unseen_baseline'] = round(test_unseen_baseline, 4)

test_sentences = test_unseen.text.to_numpy()
test_labels = keras.utils.to_categorical(test_unseen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

loss, test_unseen_accuracy = model.evaluate(test_padded, test_labels, verbose=1)
print('Phase 2 accuracy: {:5.2f}%'.format(100 * test_unseen_accuracy))
scores['test_unseen_accuracy'] = round(test_unseen_accuracy, 4)

preds = model.predict(test_padded)
df_unseen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])
df_unseen.to_csv('/home/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_unseen_text.csv', index=False)

with open('scores.json', 'w') as fp:
    json.dump(scores, fp, indent=4)
