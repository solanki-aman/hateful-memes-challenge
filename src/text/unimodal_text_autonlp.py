import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("am4nsolanki/autonlp-text-hateful-memes-36789092",
                                                           use_auth_token=False)
tokenizer = AutoTokenizer.from_pretrained("am4nsolanki/autonlp-text-hateful-memes-36789092", use_auth_token=False)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

test_seen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')
test_unseen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_unseen.csv')

test_seen = test_seen_original.copy()
test_unseen = test_unseen_original.copy()

print('Starting Phase 1 Predictions')
# Test Seen Predictions (Phase 1)
y_pred_seen = []
y_pred_proba_seen = []
for sentence in test_seen['text']:
    result = classifier(sentence)[0]
    y_pred_seen.append(result['label'])
    y_pred_proba_seen.append(result['score'])

test_seen['label_pred'] = y_pred_seen
test_seen['label_confidence'] = y_pred_proba_seen

test_seen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                 '/test_seen_autonlp.csv', index=False)

print('Starting Phase 2 Predictions')
# Test Unseen Predictions (Phase 2)
y_pred_unseen = []
y_pred_proba_unseen = []
for sentence in test_unseen['text']:
    result = classifier(sentence)[0]
    y_pred_unseen.append(result['label'])
    y_pred_proba_unseen.append(result['score'])

test_unseen['label_pred'] = y_pred_unseen
test_unseen['label_confidence'] = y_pred_proba_unseen

test_unseen.to_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/src/text/predictions'
                   '/test_unseen_autonlp.csv', index=False)
