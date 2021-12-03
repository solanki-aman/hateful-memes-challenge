import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

text_unseen_all = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/ensemble/predictions'
                              '/text_unseen_predictions.csv')
text_unseen_lstm = text_unseen_all[['label_0_lstm', 'label_1_lstm']]
text_unseen_tfidf = text_unseen_all[['label_0_tfidf', 'label_1_tfidf']]
text_unseen_autonlp = text_unseen_all[['label_0_autonlp', 'label_1_autonlp']]

image_unseen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/image/predictions'
                           '/test_unseen_image.csv')

multimodal_unseen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/image-text-concat'
                                '/predictions/test_unseen_multimodal.csv')

y_true = text_unseen_all['y_true']
baseline_accuracy = y_true.value_counts(normalize=True)[0]

# Ensembles
tfidf_image = np.mean(np.array([image_unseen, text_unseen_tfidf]), axis=0)
y_pred = np.argmax(tfidf_image, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print('Baseline Accuracy:', round(baseline_accuracy, 4))
print('Ensemble Model Accuracy:', round(accuracy, 4))
