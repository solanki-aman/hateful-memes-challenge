from tensorflow.keras.utils import Sequence
from tensorflow import keras
from utils import get_image_arrays
import numpy as np


class DataLoader(Sequence):
    def __init__(self, image_array, labels, batch_size):
        self.image_array = image_array
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_array) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_array[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_x, batch_y
