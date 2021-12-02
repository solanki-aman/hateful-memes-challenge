import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from progressbar import ProgressBar
import re
import matplotlib.image as mpimg
import tensorflow as tf
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


def get_image_predictions(image_array, model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = image_array
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

def show_image(image_id, image_path):
    image_id_dict = dict(image_id).values()
    image_id_string = list(image_id_dict)[0]
    img = mpimg.imread(image_path + image_id_string)
    plt.imshow(img)
    plt.show()