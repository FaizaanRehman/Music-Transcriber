# Polyphonic Piano Pitch and Onset Estimation #

The goal of this version is to accurately predict the onset (start times) and pitch (fundamental frequency) of piano notes given the audio of polyphonic music.

## Example ##
Here is an excerpt of Piano Sonata in A major, D.959 Mov II. Andanito (Schubert, Franz):

[Excerpt](https://user-images.githubusercontent.com/59456593/131266843-4ac316c4-d420-4e73-b7af-73429de3198b.mp4)

Here is the time-series representation of the signal:

![1733time-series](https://user-images.githubusercontent.com/59456593/131266905-3d55462b-3955-4dc2-aea2-333ff37cde1b.png)

and the mel-spectrogram respresentation:

![1733melspec](https://user-images.githubusercontent.com/59456593/131266909-e6bfd613-d685-44b5-aefc-25b0b57305b6.png)

### Results ###
A CNN was used to predict onset events given 2x128 frame samples of the mel-spectrogram image.
When an onset event is predicted for a given frame, a 1x128 vector of the mel-spectrogram image is fed into an MLP trained to estimate which notes/pitches are present in the sample.
Here are the estimations for the previous audio sample, in MIDI format:

![1733prediction](https://user-images.githubusercontent.com/59456593/131267270-218c5370-5d8d-4704-8536-2c4ab3afbd5a.png)

Compared to the actual MIDI score, the estimations were not very accurate. 

![1733midi](https://user-images.githubusercontent.com/59456593/131267097-535cbeb6-1ca0-4994-9755-0257f6d96048.png)


