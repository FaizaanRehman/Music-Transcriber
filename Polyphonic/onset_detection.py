import numpy as np

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
    onset_data = []
    for key in keys:
        X, Y = dataset[key]
        X_mel = librosa.feature.melspectrogram(X, sr=fs, hop_length=stride, n_fft=fourier_width, fmin=27, fmax=8400)
        X_mel_db = librosa.power_to_db(abs(X_mel))
        X_mel_db_T = X_mel_db.T
        previous_labels = set()
        for i in range(0, X_mel_db_T.shape[0]-2, 2):
            labels = Y[i*stride]   
            # add image of mel frequencies (data) 3x128 pixels
            mel_data.append(X_mel_db_T[i:i+2])
            # get notes present in current frame that were not present in previous frame, indicates onset
            differences = labels - previous_labels
            if (len(differences)):
                onset_data.append(1)
            else: 
                onset_data.append(0)
            previous_labels = labels
    # save data for future use
    with open('onset_data.npy', 'wb') as file:
        np.save(file, np.asarray(mel_data))
        np.save(file, np.asarray(onset_data))

def main():
    # obtain all keys with piano solos
    if not os.path.exists('solo_keys.npy'):
        keys = retrieveKeys()
        with open('solo_keys.npy', 'wb') as file:
            np.save(file, np.asarray(keys))
    with open('solo_keys.npy', 'rb') as file:
        keys = np.load(file, allow_pickle=True, encoding='latin1')
    
    # Load custom_data for training
    if not os.path.exists('onset_data.npy'):
        prepareCustomData(keys)
    with open('onset_data.npy', 'rb') as file:
        X = np.load(file, allow_pickle=True, encoding='latin1')
        Y = np.load(file, allow_pickle=True, encoding='latin1')

    print(type(X))
    print(type(Y))
    print(X.shape)
    print(Y.shape)
    print(type(X[0]))
    print(X[0].shape)
    print(type(Y[0]))
    X = X[..., np.newaxis]
    print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3, shuffle=True)

    model = keras.Sequential([
        keras.layers.Conv2D(filters=4, kernel_size=(3,3), activation='relu', input_shape=(2,128,1), padding='same'),
        keras.layers.MaxPool2D((2,2), padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_absolute_error'])

    history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), batch_size=128)
    model.save('onset_detection_model')

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

    print(history.history.keys())
    acc = history.history['mean_absolute_error']
    val_acc = history.history['mean_absolute_error']
    plt.plot(epochs, acc, 'y', label='Training error')
    plt.plot(epochs, val_acc, 'r', label='Validation error')
    plt.title('Training and validation error')
    plt.xlabel('Epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()