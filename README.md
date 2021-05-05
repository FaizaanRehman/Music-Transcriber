# Music Transcriber

This project attempts to transcribe piano music into its symbolic representation.

### Heading 3 ###

General overview of design/workflow:
- Accept an audio file contain piano music (eg: .mp3, .ogg, .wav)
- Preprocess audio signals (segmentation, fourier transform, spectogram)
- Pass signals into trained neural network and retrieve predictions
- Format output into interpretable info (MIDI number corresponding to note)
