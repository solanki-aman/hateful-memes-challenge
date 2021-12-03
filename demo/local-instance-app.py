import streamlit as st
import pandas as pd
import numpy as np
import pickle

from utils import get_image_arrays, get_image_predictions, show_image

st.title('Hateful Memes Classification')

image_path = '/Users/amansolanki/datasets/hateful-memes-images/'
test_seen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
TFLITE_FILE_PATH = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/image_model.tflite'
test_seen = test_seen_original.sample(1)
y_true = test_seen['label']
image_id = test_seen['image_id']
text = test_seen['text']

image_id_dict = dict(image_id).values()
image_id_string = list(image_id_dict)[0]
st.write('Meme:')
st.image(image_path+image_id_string)

# Image Unimodel
image_array = get_image_arrays(image_id, image_path)
image_prediction = get_image_predictions(image_array, TFLITE_FILE_PATH)
y_pred_image = np.argmax(image_prediction, axis=1)
print('Image Prediction Probabilities:')
print(image_prediction)

# TFIDF Model
model = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/tfidf_model.pickle'
vectorizer = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/tfidf_vectorizer.pickle'
tfidf_model = pickle.load(open(model, 'rb'))
tfidf_vectorizer = pickle.load(open(vectorizer, 'rb'))
transformed_text = tfidf_vectorizer.transform(text)
text_prediction = tfidf_model.predict_proba(transformed_text)
y_pred_text = np.argmax(text_prediction, axis=1)
print('Text Prediction Probabilities:')
print(text_prediction)

# Ensemble Probabilities
ensemble_prediction = np.mean(np.array([image_prediction, text_prediction]), axis=0)
y_pred_ensemble = np.argmax(ensemble_prediction, axis=1)
print(ensemble_prediction)

# StreamLit Display
st.write('Image Model Predictions:')
st.write(np.round(np.array(image_prediction), 4))

st.write('Text Model Predictions:')
st.write(np.round(np.array(text_prediction), 4))

st.write('Ensemble Model Predictions:')
st.write(np.round(np.array(ensemble_prediction), 4))

true_label = list(dict(y_true).values())[0]
predicted_label = y_pred_ensemble[0]

st.write('True Label', true_label)
st.write('Predicted Label', predicted_label)
st.write('0: non-hateful, 1: hateful')

st.button('Random Meme')

