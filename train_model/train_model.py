import tensorflow.keras.models as sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
import numpy as np
import json

TIME_PERIODS = 20
num_sensors = 39

model = sequential.Sequential()

model.add(Conv1D(32, 5, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(16, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))

print(model.summary())

x_train = []
y_train = []

with open("./x_train.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        x_train.append(data)

with open("./y_train.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        y_train.append(data)


x_train = np.array(x_train).astype(float)
y_train = np.array(y_train).astype(float)

print(np.shape(x_train), np.shape(y_train))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
BATCH_SIZE = 128
EPOCHS = 200
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
model.save('bad3.h5')
