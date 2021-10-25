import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
class Model:
    def __init__(self):
        self.model = tf.keras.models.load_model('./model.h5')
        self.data = np.array([0.0]*20*39)
        self.data = self.data.reshape(1, 20, 39)

    def append(self, frame_data):
        #print(np.shape(frame_data))
        self.data = self.data[0][1:21][:]
        self.data = np.append(self.data, frame_data.split(','))
        self.data = self.data.reshape(1, 20, 39)

    def predict(self,DD):
        results = np.round(self.model.predict(DD.astype(float)), 3)
        return np.argmax(results) + 1
