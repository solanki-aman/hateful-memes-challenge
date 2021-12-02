import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from utils import transform_autonlp_df, merge_predictions

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

test_seen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_unseen = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')

text_path = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions/'
test_seen_autonlp_original = pd.read_csv(text_path+'test_seen_autonlp.csv')
test_unseen_autonlp_original = pd.read_csv(text_path+'test_unseen_autonlp.csv')

test_seen_text = pd.read_csv(text_path+'test_seen_text.csv')
test_unseen_text = pd.read_csv(text_path+'test_unseen_text.csv')

test_seen_text_tfidf = pd.read_csv(text_path+'test_seen_text_tfidf.csv')
test_unseen_text_tfidf = pd.read_csv(text_path+'test_unseen_text_tfidf.csv')

# Transform AutoNLP Results
test_seen_autonlp = transform_autonlp_df(test_seen_autonlp_original)
test_unseen_autonlp = transform_autonlp_df(test_unseen_autonlp_original)

# Merge Predictions
test_seen_predictions = merge_predictions(test_seen_text, test_seen_text_tfidf, test_seen_autonlp)
test_unseen_predictions = merge_predictions(test_unseen_text, test_unseen_text_tfidf, test_unseen_autonlp)

# Add true labels to predictions
test_seen_predictions['y_true'] = test_seen['label']
test_unseen_predictions['y_true'] = test_unseen['label']

prediction_save_path = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/ensemble/predictions/'
test_seen_predictions.to_csv(prediction_save_path+'text_seen_predictions.csv', index=False)
test_unseen_predictions.to_csv(prediction_save_path+'text_unseen_predictions.csv', index=False)