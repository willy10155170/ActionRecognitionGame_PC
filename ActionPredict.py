import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
class Model:
    def __init__(self):
        self.model = tf.keras.models.load_model('./model.h5')

    def predict(self, predict_data):
        predict_data = np.array(predict_data)
        predict_data = predict_data.reshape(1, 20, 39)
        result = self.model.predict(predict_data).reshape(4, )
        #print("punch = {:0.2f}, defense = {:0.2f}, skill1 = {:0.2f}, skill2 = {:0.2f}".format(result[0], result[1], result[2], result[3]))
        return np.argmax(result) + 1
