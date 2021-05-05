# Music Transcriber #

This project attempts to transcribe piano music into its symbolic representation. For more information, please visit: https://en.wikipedia.org/wiki/Transcription_(music)#Automatic_music_transcription

Currently, the program seeks to determine the pitch of the notes in a piano audio sample.

### General Design/Workflow: ###
- Accept an audio file contain piano music (eg: .mp3, .ogg, .wav)
- Preprocess audio signals (segmentation, fourier transform, spectogram)
- Pass signals into trained neural network and retrieve predictions
- Format output into interpretable info (MIDI number corresponding to note)

## Example ##
Here is a processed audio sample of the piano note C5 (MIDI note 52):

![Figure_1](https://user-images.githubusercontent.com/59456593/117212926-f6c37080-adc8-11eb-8292-c625967b74ed.png)

The Neural Network predicted the note B4 (MIDI note 51) over most of the frames of the audio, which is one semitone below the intended result:

![Figure_2](https://user-images.githubusercontent.com/59456593/117213324-76513f80-adc9-11eb-96e6-4a5aae32c364.png)

# Future Goals #
To improve the precision/accuracy of the Neural Network:
- Supply more training data with more variety of piano sounds and notes
- Experiment with other Network types (CNN, RNN, LSTM)
- Experiment with other audio processing methods (Contant-Q transform, Chroma-feature)


## Resources Used ##
The training data for the neural network was obtained from: https://freesound.org/people/TEDAgame/packs/25405/?page=6#sound
For processing the audio and structuring the neural network, tutorials and examples from 'The Sound of AI' were followed: https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ
