import json
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scores = dict()
config = dict()
df = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/train.csv')

# Features and Labels
text = df['text']
label = df['label']
config['_num_labels'] = label.nunique()

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(text, label, test_size=0.1, random_state=42, shuffle=True)
validation_set_baseline = y_val.value_counts(normalize=True)[0]
scores['validation_baseline'] = round(validation_set_baseline, 4)

# Transform Train and Test Data
vectorizer = TfidfVectorizer(analyzer='word')
X_train_transformed_temp = vectorizer.fit_transform(X_train)
X_val_transformed_temp = vectorizer.transform(X_val)

X_train_transformed = pd.DataFrame(X_train_transformed_temp.toarray(), columns=vectorizer.get_feature_names_out())
X_val_transformed = pd.DataFrame(X_val_transformed_temp.toarray(), columns=vectorizer.get_feature_names_out())

# Logistic Regression Model
classifier = LGBMClassifier(objective='binary',
                            boosting_type='dart',
                            importance_type='gain',
                            n_estimators=20,
                            random_state=42,
                            n_jobs=-1)

model = classifier.fit(X_train_transformed, y_train)

# Score
y_pred = model.predict(X_val_transformed)
y_pred_proba = model.predict_proba(X_val_transformed)
validation_accuracy = accuracy_score(y_val, y_pred)
scores['validation_set_accuracy'] = round(validation_accuracy, 4)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 1 Predictions')
test_seen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_seen = test_seen_original.copy()

test_seen_text = test_seen['text']
test_seen_labels = test_seen['label']
test_seen_baseline = test_seen_labels.value_counts(normalize=True)[0]
scores['test_seen_baseline'] = round(test_seen_baseline, 4)

test_seen_transformed_temp = vectorizer.transform(test_seen_text)
test_seen_transformed = pd.DataFrame(test_seen_transformed_temp.toarray(), columns=vectorizer.get_feature_names_out())
y_pred_seen = model.predict(test_seen_transformed)
y_pred_proba_seen = model.predict_proba(test_seen_transformed)
test_seen_accuracy = accuracy_score(test_seen_labels, y_pred_seen)
scores['test_seen_accuracy'] = round(test_seen_accuracy, 4)

df_seen = pd.DataFrame(y_pred_proba_seen, columns=['label_0_confidence', 'label_1_confidence'])

df_seen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
               '/test_seen_text_tfidf.csv', index=False)

# --------------------------------------------------------------------------------------------------------------
print('Starting Phase 2 Predictions')
test_unseen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')
test_unseen = test_unseen_original.copy()

test_unseen_text = test_unseen['text']
test_unseen_labels = test_unseen['label']
test_unseen_baseline = test_unseen_labels.value_counts(normalize=True)[0]
scores['test_unseen_baseline'] = round(test_unseen_baseline, 4)

test_unseen_transformed_temp = vectorizer.transform(test_unseen_text)
test_unseen_transformed = pd.DataFrame(test_unseen_transformed_temp.toarray(),
                                       columns=vectorizer.get_feature_names_out())
y_pred_unseen = model.predict(test_unseen_transformed)
y_pred_proba_unseen = model.predict_proba(test_unseen_transformed)
test_unseen_accuracy = accuracy_score(test_unseen_labels, y_pred_unseen)
scores['test_unseen_accuracy'] = round(test_unseen_accuracy, 4)

df_unseen = pd.DataFrame(y_pred_proba_unseen, columns=['label_0_confidence', 'label_1_confidence'])

df_unseen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_unseen_text_tfidf.csv', index=False)

with open('scores_tfidf.json', 'w') as fp:
    json.dump(scores, fp, indent=4)
