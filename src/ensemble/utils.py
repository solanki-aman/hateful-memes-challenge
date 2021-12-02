import pandas as pd


def transform_autonlp_df(df: pd.DataFrame):
    # Filter DataFrame
    df_0_seen = df[df['label_pred'] == 0]
    df_1_seen = df[df['label_pred'] == 1]

    # Clean Column Names
    df_0_seen.columns = ['image_id', 'text', 'label', 'label_pred', 'label_0_confidence']
    df_1_seen.columns = ['image_id', 'text', 'label', 'label_pred', 'label_1_confidence']

    # Calculate Probabilities for label
    df_0_seen['label_1_confidence'] = 1 - df_0_seen['label_0_confidence']
    df_1_seen['label_0_confidence'] = 1 - df_1_seen['label_1_confidence']

    # Concatenate DataFrames
    df_concat = pd.concat([df_0_seen, df_1_seen], axis=0)
    df_concat = df_concat.sort_index()

    # Return Probabilities Columns
    df_predict_proba = df_concat[['label_0_confidence', 'label_1_confidence']]

    return df_predict_proba


def merge_predictions(df_lstm: pd.DataFrame, df_tfidf: pd.DataFrame, df_autonlp: pd.DataFrame):
    df_lstm.columns = ['label_0_lstm', 'label_1_lstm']
    df_tfidf.columns = ['label_0_tfidf', 'label_1_tfidf']
    df_autonlp.columns = ['label_0_autonlp', 'label_1_autonlp']

    test_seen_predictions = pd.concat([df_lstm, df_tfidf, df_autonlp], axis=1)

    # test_seen_predictions['lstm'] = np.argmax(df_lstm.to_numpy(), axis=1)
    # test_seen_predictions['tfidf'] = np.argmax(df_tfidf.to_numpy(), axis=1)
    # test_seen_predictions['autonlp'] = np.argmax(df_autonlp.to_numpy(), axis=1)

    return test_seen_predictions
