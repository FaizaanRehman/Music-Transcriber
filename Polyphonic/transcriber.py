import numpy as np
import matplotlib.pyplot as plt

import librosa
import tensorflow.keras as keras

fs = 22050 # sampling frequency in hertz
fourier_width = 2048 # Each FFT operates on 2048 sample frames
stride = 1024    # windows are 1024 samples wide 

# retrieve audio file to transcribe
X, _ = librosa.load('musicnet/train_data/1733.wav', sr=fs)
print(type(X))
X_mel = librosa.feature.melspectrogram(X, sr=fs, n_fft=fourier_width, hop_length=stride, fmin=27, fmax=8400)
X_mel_T = X_mel.T

onset_model = keras.models.load_model('onset_detection_model')
pitch_model = keras.models.load_model('pitch_detection_model')

predictions = np.zeros(X_mel_T.shape)
onsets = 0
for i in range(0, 300):
    prediction = onset_model.predict(X_mel_T[i:i+2].reshape(1, 2, 128, 1))[0,0] 
    if round(prediction+0.3)==1:        # add some bias (0.3) to encourage more predictions
        onsets += 1
        for j in range(i, i+5):
            predictions[j] = pitch_model.predict(X_mel_T[j].reshape(1,128))

plt.imshow(np.round(predictions[0:300].T), cmap='hot', interpolation='nearest', aspect='auto')
plt.show()
