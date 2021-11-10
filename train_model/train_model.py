import tensorflow as tf
import tensorflow.keras.models as sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
import numpy as np
import json

TIME_PERIODS = 20
num_sensors = 39

model = sequential.Sequential()
# model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(20, 39)))
# model.add(Conv1D(32, 5, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# model.add(MaxPooling1D(2))
# model.add(Dropout(0.1))
# model.add(Conv1D(16, 1, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Dropout(0.1))
# model.add(Conv1D(16, 1, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))


model.add(Conv1D(32, 5, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(16, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.1))
model.add(Dense(4, activation='softmax'))


# model.add(Conv1D(16,5, activation = 'relu', input_shape= (20,39)))
# model.add(Conv1D(16,5, activation = 'relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(32,5, activation = 'relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.1))
# model.add(Dense(4, activation='softmax'))
print(model.summary())

x_train = []
y_train = []

with open("./test_data/x_train1.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        x_train.append(data)

with open("./test_data/y_train1.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        y_train.append(data)


x_train = np.array(x_train).astype(float)
y_train = np.array(y_train).astype(float)

print(np.shape(x_train), np.shape(y_train))


#callbacks_list = [tf.keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
BATCH_SIZE = 3924
EPOCHS = 200
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
model.save('model.h5')
