import time
import os
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
df = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/train.csv')

EPOCHS = 50

# Features and Labels
text = df['text']
label = df['label']

config['_num_labels'] = label.nunique()

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(text, label, test_size=0.2, random_state=42, shuffle=True)

# Clean Training Data
temp_train = X_train.map(remove_punctuations)
X_train_clean = temp_train.map(remove_stopwords)
# Clean Validation Data
temp_test = X_val.map(remove_punctuations)
X_val_clean = temp_test.map(remove_stopwords)

# Build Vocabulary
word_set = build_word_set(X_train_clean.to_list())
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
tokenizer.fit_on_texts(X_train_clean.to_list())

# Transform Sentence Words to Tokens
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# Create Padded Train and Validation Sequences for Model Training
sentence_max_length = max([len(sentence.split()) for sentence in train_sentences])  # 47
max_length = 20
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
config['max_length'] = max_length
config["padding"] = "max_length"

# Glove Word Embeddings
# Embedding Transfer Learning
path_to_glove_file = '/Users/amansolanki/datasets/glove.6B.100d.txt'

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

num_tokens = num_unique_words + 2
embedding_dim = 100
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
model.add(layers.LSTM(8, dropout=0.2, kernel_regularizer='l2'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(layers.Dropout(0.8))
model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer='l2'))
model.add(layers.Dropout(0.8))
model.add(keras.layers.Dense(6, activation='relu', kernel_regularizer='l2' ))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(2, activation="sigmoid"))

model.summary()

loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=1e-08)
# optimizer = keras.optimizers.SGD(learning_rate=0.1)
# optimizer = keras.optimizers.Adadelta(learning_rate=1e-04, rho=0.95, epsilon=1e-07)
EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')

text_unimodal = model.fit(train_padded,
                          train_labels,
                          epochs=EPOCHS,
                          validation_data=(val_padded, val_labels),
                          callbacks=[EarlyStoppingCallback],
                          verbose=1)

show_interactive_performance_plot(text_unimodal, 'text_unimodal_accuracy', 'accuracy', 'val_accuracy')
show_interactive_performance_plot(text_unimodal, 'text_unimodal_loss', 'loss', 'val_loss')


end_time = time.time()
time_taken = end_time - start_time
time_taken_minutes = round(time_taken/60, 3)

loss, validation_set_accuracy = model.evaluate(val_padded, val_labels, verbose=1)
print('Validation accuracy: {:5.2f}%'.format(100 * validation_set_accuracy))
scores['validation_set_accuracy'] = round(validation_set_accuracy, 4)

config['training_time'] = time_taken_minutes

with open('config.json', 'w') as fp:
    json.dump(config, fp, indent=4)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 1 Predictions')
test_seen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen_lstm.csv')
test_seen = test_seen_original.copy()

test_seen['text'] = test_seen.text.map(remove_punctuations)
test_seen['text'] = test_seen.text.map(remove_stopwords)

text = test_seen['text']
labels = test_seen['label']

test_sentences = test_seen.text.to_numpy()
# test_labels = test_unseen.label.to_numpy()
test_labels = keras.utils.to_categorical(test_seen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

max_length = 20

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

loss, test_seen_accuracy = model.evaluate(test_padded, test_labels, verbose=1)
print('Phase 1 accuracy: {:5.2f}%'.format(100 * test_seen_accuracy))
scores['test_seen_accuracy'] = round(test_seen_accuracy, 4)

preds = model.predict(test_padded)
df_seen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])

df_seen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_seen_predications.csv', index=False)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 2 Predictions')
test_unseen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen_lstm.csv')
test_unseen = test_unseen_original.copy()

test_unseen['text'] = test_unseen.text.map(remove_punctuations)
test_unseen['text'] = test_unseen.text.map(remove_stopwords)

text = test_unseen['text']
labels = test_unseen['label']

test_sentences = test_unseen.text.to_numpy()
# test_labels = test_unseen.label.to_numpy()
test_labels = keras.utils.to_categorical(test_unseen['label'], 2)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

max_length = 20

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

loss, test_unseen_accuracy = model.evaluate(test_padded, test_labels, verbose=1)
print('Phase 2 accuracy: {:5.2f}%'.format(100 * test_unseen_accuracy))
scores['test_unseen_accuracy'] = round(test_unseen_accuracy, 4)

preds = model.predict(test_padded)
df_unseen = pd.DataFrame(preds, columns=['label_0_confidence', 'label_1_confidence'])
df_unseen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_unseen_predications.csv', index=False)

with open('scores.json', 'w') as fp:
    json.dump(scores, fp, indent=4)
