# Polyphonic Piano Pitch and Onset Estimation #

The goal of this version is to accurately predict the onset (start times) and pitch (fundamental frequency) of piano notes given the audio of polyphonic music.

## Example ##
Here is an excerpt of Piano Sonata in A major, D.959 Mov II. Andanito (Schubert, Franz):

[Excerpt](https://user-images.githubusercontent.com/59456593/131266843-4ac316c4-d420-4e73-b7af-73429de3198b.mp4)

Here is the time-series representation of the signal:

![1733time-series](https://user-images.githubusercontent.com/59456593/131267351-bb2aea85-501a-4bfc-a814-582cfe98410b.png)

and the mel-spectrogram respresentation:

![1733melspec](https://user-images.githubusercontent.com/59456593/131267358-ec94e98b-3a5d-4388-af1a-608684b82f2b.png)

### Results ###
A CNN was used to predict onset events given 2x128 frame samples of the mel-spectrogram image.
When an onset event is predicted for a given frame, a 1x128 vector of the mel-spectrogram image is fed into an MLP trained to estimate which notes/pitches are present in the sample.
Here are the estimations for the previous audio sample, in MIDI format:

![1733prediction](https://user-images.githubusercontent.com/59456593/131267270-218c5370-5d8d-4704-8536-2c4ab3afbd5a.png)

Compared to the actual MIDI score, the estimations were not very accurate. 

![1733midi](https://user-images.githubusercontent.com/59456593/131267363-95e662dc-450a-4ca8-905b-bcc7e4f9a7da.png)

### Potential Causes of Innaccuracy ###
The models used for this project were constructed with a focus on simplicity. 
Here are the training results for the pitch estimator model:

![loss](https://user-images.githubusercontent.com/59456593/131267622-d627df91-0b56-476e-b3c9-bbd9d418988a.png)
![accuracy](https://user-images.githubusercontent.com/59456593/131267628-04ed5dc2-426a-4c54-a114-e3a534ef99f6.png)

Here are the training results for the onset predictor model:

![loss_onset](https://user-images.githubusercontent.com/59456593/131267953-85995d52-bf39-4acc-b9fd-c2a91835f0cd.png)
![onset_error](https://user-images.githubusercontent.com/59456593/131267959-2c297536-0760-4d7d-8337-d3b6e442e493.png)

Here are the 4 convolution filters that were generated in this model:

![convfilters](https://user-images.githubusercontent.com/59456593/131404194-3ee900eb-65e2-44a9-9817-eee497ab7ff2.png)

Ultimately it does not seem like the current state of these models are appropriate for the prediction tasks of this project.

# Future Improvements #
To improve the accuracy and correctness of music transcription:
- Obtain more features/information from the audio data, the mel-spectrogram alone does not seem to provide enough information to correctly estimate
- Modify/customize the training data such that the information is more useful (eg. only sample audio near onset times) 
- Experiment with LSTM networks, as they can make predictions based on a sequence of data rather than individual slices, which may work better with musical data.


## Resources Used ##
All audio data and labels used in this project were obtained from the MusicNet dataset: https://homes.cs.washington.edu/~thickstn/musicnet.html
