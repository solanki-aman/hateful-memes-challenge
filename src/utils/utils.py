import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from progressbar import ProgressBar
import re
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def get_image_arrays(image_column, image_path):
    progressBar = ProgressBar()
    X = []

    for image_id in progressBar(image_column.values):
        image = load_img(image_path + image_id, target_size=(224, 224))
        image_array = img_to_array(image)

        X.append(image_array)

    X_array = np.asarray(X, dtype='float32')
    X_array /= 255.

    return X_array


def show_images(X, rows, columns):
    fig = plt.figure(figsize=(10, 10))
    columns = columns
    rows = rows
    for i in range(1, columns * rows + 1):
        total_images = X.shape[0]
        random_number = np.random.randint(total_images - 1)
        image_id = random_number
        img = X[image_id]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.tight_layout()
    plt.show()


def show_interactive_performance_plot(model, title: str, training_metric: str, validation_metric: str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(1, len(model.history[training_metric]) + 1),
                             y=model.history[training_metric],
                             mode='lines+markers',
                             name='TRAINING'))

    fig.add_trace(go.Scatter(x=np.arange(1, len(model.history[validation_metric]) + 1),
                             y=model.history[validation_metric],
                             mode='lines+markers',
                             name='VALIDATION'))

    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},

        xaxis_title="EPOCHS",
        yaxis_title=training_metric.split('_')[-1].upper(),
        font=dict(
            family="Arial",
            size=11,
            color="Black"
        )
    )
    path = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/plots/'
    filename = path + title + '.jpeg'
    fig.write_image(filename, width=1920, height=1080, scale=2.5)

    fig.show()

