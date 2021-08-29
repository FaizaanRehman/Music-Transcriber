import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

from notes import midi_array

SAMPLE_RATE = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 2048
NUM_BINS = 88
MIN_FREQ = 27.5 # in Hz

DATASET_PATH = "Sample Notes/"
JSON_PATH = "data.json"

def prepare_data():
    # dictionary to store data
    data = {
        'note': [],
        'mel_coeffs': [],
        'labels': []
    }

    # loop through each note and its sound files
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        if dirpath == DATASET_PATH:
            continue
        
        # add note name to data
        note = dirpath.split('/')[-1]
        data['note'].append(note)

        # load each sound file
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, duration=3.0)

            # compute mel-spectrogram of sound wave
            signal_mel = librosa.feature.melspectrogram(signal, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

            # store data in dictionary
            signal_mel = signal_mel.T
            for j in range(5, len(signal_mel)):
                data['mel_coeffs'].append(signal_mel[j].tolist())
                data['labels'].append(i-1)

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)    
    

def load_data():
    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)
        
    inputs = np.array(data['mel_coeffs'])
    targets = np.array(data['labels'])

    return inputs, targets


def build_network(inputs, targets, save=True):
    
    # split data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    # build neural network model
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(128,)),

        # 1st dense layer
        keras.layers.Dense(256, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(128, activation='relu'),

        # output layer
        keras.layers.Dense(88, activation='softmax')
    ])

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train network
    model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32)

    # save model
    if (save):
        model.save('transcriber_model')

def initialize_app():
    # Prepare and save training data
    prepare_data()

    # Load training data
    inputs, targets = load_data()

    # Build and train network, and save
    build_network(inputs, targets)

if __name__ == "__main__":

    # initialize_app() is only needed for the first run
    # For subsequent runs, comment out to reduce runtime
    # initialize_app()

    # Load network 
    model = keras.models.load_model('transcriber_model')
   
    # Load File
    signal, sr = librosa.load("Sample Notes/C5/448548__tedagame__c5.ogg", duration=3.0)

    # Compute mel-spectrogram of sound wave
    signal_mel = librosa.feature.melspectrogram(signal, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    # Use model to predict
    signal_mel = signal_mel.T
    predictions = model.predict(signal_mel)

    # Rest of code is for displaying the plots
    indices = np.argmax(predictions, axis=1)

    midi_notes = []
    
    for i in range(len(indices)):
        midi_notes.append(midi_array[indices[i]])

    midi_notes = np.array(midi_notes)

    signal_mel_db = librosa.power_to_db(abs(signal_mel))

    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_mel_db.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    plt.step(range(len(midi_notes)), midi_notes)
    plt.show()

    


    
    
    
    







