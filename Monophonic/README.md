# Monophonic Piano Pitch Estimation #

The goal of this attempt is to accurately estimate the pitch of a piano note given its audio sample.

### General Design/Workflow: ###
- Accept an audio file contain piano music (eg: .mp3, .ogg, .wav)
- Preprocess audio signals (segmentation, fourier transform, spectogram)
- Pass signals into trained neural network and retrieve predictions
- Format output into interpretable info (MIDI number corresponding to note)

## Example ##
Here is the audio sample of the piano note C5 (key number 52):

Here is a processed audio sample of the piano note C5 (key number 52) in its Mel Spectrogram respresentation:

![C5melspec](https://user-images.githubusercontent.com/59456593/131265878-b96be13a-81bc-47e7-93e6-45849279a09c.png)

The Neural Network predicted the note G3 (key number 35) over most of the frames of the audio, which is much lower than the expected note. Over the first half of the audio sequence, the prediction is quite strong. As the audio begins to fade, the prediction becomes weaker and it begins to consider other notes.

![C5predic](https://user-images.githubusercontent.com/59456593/131266038-5df316af-b71c-402b-ac0e-f6c11aaec19b.png)

# Future Goals #
To improve the precision/accuracy of the Neural Network:
- Supply more training data with more variety of piano sounds and notes
- Experiment with other Network types (CNN, RNN, LSTM)
- Experiment with other audio processing methods (Contant-Q transform, Chroma-feature)


## Resources Used ##
- The training data for the neural network was obtained from: https://freesound.org/people/TEDAgame/packs/25405/?page=6#sound
- For processing the audio and structuring the neural network, tutorials and examples from 'The Sound of AI' were followed: https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ
