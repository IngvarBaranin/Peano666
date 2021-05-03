import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import json

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
# Dynamically grabs data for the model, since the whole thing wouldn't fit into memory all at once.
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, training_data, batch_size, num_classes, shuffle=True):
        self.training_data = training_data
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

    def __len__(self):
        # returns the number of batches
        return int(np.floor(len(training_data) / self.batch_size))

    def __getitem__(self, idx):
        X = [i.split(", ")[0].split(" ") for i in self.training_data[idx * self.batch_size:(idx + 1) * self.batch_size]]
        X = [[int(integer) for integer in integers] for integers in X]
        y = [i.split(", ")[1] for i in self.training_data[idx * self.batch_size:(idx + 1) * self.batch_size]]
        y = [int(integer) for integer in y]

        return to_categorical(X, num_classes=self.num_classes), to_categorical(y, num_classes=self.num_classes)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.training_data)


# Load vocabulary
vocabulary = {token: int(token_int) for token, token_int in json.load(open("./dictionary.json")).items()}

# Count the lines in training_data
with open("./training_data_preprocessed.txt") as f:
    training_data = f.read().splitlines()

# Instantiate generator with batch size 512, shuffling the data each epoch
training_generator = CustomDataset(training_data, 512, len(vocabulary), True)

# Config to save model after every epoch if it is better than all previous ones in terms of minimal loss
filepath = "../models/simple/SimpleLSTM-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_weights_only=False,
    save_best_only=True,
    mode='min'
)

if __name__ == '__main__':
    model = Sequential()
    model.add(LSTM(256, input_shape=(100, len(vocabulary),)))
    model.add(Dropout(0.2))
    model.add(Dense(len(vocabulary), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(training_generator,
              epochs=100,
              use_multiprocessing=True,
              callbacks=[checkpoint],
              workers=6)