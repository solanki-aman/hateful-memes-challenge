import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

text_seen_all = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/ensemble/predictions'
                            '/text_seen_predictions.csv')
text_seen_lstm = text_seen_all[['label_0_lstm', 'label_1_lstm']]
text_seen_tfidf = text_seen_all[['label_0_tfidf', 'label_1_tfidf']]
text_seen_autonlp = text_seen_all[['label_0_autonlp', 'label_1_autonlp']]

image_seen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/predictions'
                         '/test_seen_image.csv')

multimodal_seen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/image-text-concat'
                              '/predictions/test_seen_multimodal.csv')

y_true = text_seen_all['y_true']
baseline_accuracy = y_true.value_counts(normalize=True)[0]

# Ensembles
tfidf_image = np.mean(np.array([image_seen, text_seen_tfidf]), axis=0)
y_pred = np.argmax(tfidf_image, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print('Baseline Accuracy:', round(baseline_accuracy, 4))
print('Ensemble Model Accuracy:', round(accuracy, 4))