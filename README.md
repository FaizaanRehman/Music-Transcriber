# Music Transcriber

This project attempts to transcribe piano music into its symbolic representation.

General overview of design/workflow:
- Accept an audio file contain piano music (eg: .mp3, .ogg, .wav)
- Preprocess audio signals (Fourier Transform, Spectograms representation)
- Pass signals into trained neural network and retrieve predictions
- Format output into interpretable info
