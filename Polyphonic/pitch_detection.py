import numpy as np
from intervaltree import IntervalTree

import librosa
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

import os.path
import csv

# Retrieve 'solo piano' song keys from Musicnet dataset
def retrieveKeys():
    song_keys = []
    with open('musicnet/musicnet_metadata.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (row[4] == 'Solo Piano'):
                song_keys.append(row[0])
    return song_keys

# Prepare/Modify Musicnet dataset for training model
def prepareCustomData(keys):
    fs = 22050  # sampling frequency in Hz
    fourier_width = 2048 # Each FFT operates on 2048 sample frames
    stride = 1024    # windows are 1024 samples wide 

    dataset = np.load(open('musicnet/musicnet.npz', 'rb'), allow_pickle=True, encoding='latin1')
    mel_data = []
    note_data = []
    for key in keys:
        X, Y = dataset[key]
        X_mel = librosa.feature.melspectrogram(X, sr=fs, hop_length=stride, n_fft=fourier_width)
        X_mel_db = librosa.power_to_db(X_mel)
        X_mel_db_T = X_mel_db.T
        for i in range(X_mel_db_T.shape[0]):
            labels = Y[i*stride]
            notes = np.zeros(128,)
            for label in labels:
                # mark the note as 1 if its played, rest will be zero
                notes[label.data[1]] = 1
            # add vector of mel frequencies (data) with notes played at that time (label)
            mel_data.append(X_mel_db_T[i])
            note_data.append(notes)
    # save data for future use
    with open('pitch_data.npy', 'wb') as file:
        np.save(file, np.asarray(mel_data))
        np.save(file, np.asarray(note_data))

def main():
    # obtain all keys with piano solos
    if not os.path.exists('solo_keys.npy'):
        keys = retrieveKeys()
        with open('solo_keys.npy', 'wb') as file:
            np.save(file, np.asarray(keys))
    with open('solo_keys.npy', 'rb') as file:
        keys = np.load(file, allow_pickle=True, encoding='latin1')

    # Load custom data for training
    if not os.path.exists('pitch_data.npy'):
        prepareCustomData(keys)
    with open('pitch_data.npy', 'rb') as file:
        X = np.load(file, allow_pickle=True, encoding='latin1')
        Y = np.load(file, allow_pickle=True, encoding='latin1')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3, shuffle=True)

    model = keras.Sequential([
        
        # input layer
        keras.layers.Flatten(input_shape=(128,)),

        # 1st dense layer
        keras.layers.Dense(128, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(128, activation='relu'),

        # output layer
        keras.layers.Dense(128, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, epochs=15, validation_data=(X_test, Y_test), batch_size=512)
    model.save('pitch_detection_model')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()