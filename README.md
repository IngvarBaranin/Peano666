# Peano666: Music Generation using RNN- and Markov-Based approaches

---

## 1. Preface

### Data

This project generates music using MIDI files encoded into text. The MIDI files are from the [piano-midi](http://www.piano-midi.de) website, which mostly consists of music from the romantic era, with a bit of classicism and baroque.

### Encoding

While this project explores two different generative processes, the encoding from which these processes are based on, is the same. Using [music21](https://web.mit.edu/music21/), MIDI files are converted into text: a *musical language*, if you will, of 208 tokens. A majority of these are related to the starting and stopping of notes, while others define pauses, velocity and tempo. 

To keep the vocabulary of the language simple, some rarely occurring (<100 occurrences) pause tokens were replaced with a frequently (>=100) seen pause token - this was done with pause length similarity in mind. Also, since the original MIDI files were very expressive, a bunch of unique velocity and tempo values were remapped to just a few dozen tokens. If this had not been done, the project would mostly be about generating tempo and velocity markings, not generating music.

### Decoding

Using music21, converting the text back to MIDI is quite straightforward: just use the text tokens to add appropriate elements to a MIDI stream. The encoding/decoding code can be inspected in [midi_utils.ipynb](https://github.com/IngvarBaranin/Peano666/blob/main/notebooks/midi_utils.ipynb).

## 2. RNN (LSTM) approach 

### Training input & target

Using a sliding window algorithm, the entirety of encoded data was iterated over to grab 100-token long sequences (input) and a single subsequent token (target). This resulted in about 2 million training samples. During training, the tokens were onehotted to make the samples more training-friendly. 

### Training

The LSTM approach uses TensorFlow's LSTM implementation. There are three LSTM layers with 256 units each. In hopes of avoiding overfitting, each LSTM layer is also followed by a dropout layer, with a dropout rate of 0.2. 

Quite a few initial training attempts stopped learning after a while. This was somewhat mended by using gradient clipping.

The best model reached a loss of 0.9498 and is [included in this repo](https://github.com/IngvarBaranin/Peano666/blob/main/best_model.hdf5).

### Generating

Initially, generating music using the trained model requires some input. We just create a 100 element long list of random tokens within our *musical language*. The output is constantly added to the input, so before long, the input is no longer randomly generated.  

## 3. Markov

## 4. Evaluation

## 5. Requirements

- Jupyter (or whatever you use to handle .ipynb files)
- Music21 (pip install music21)
- Selenium* (pip install selenium, with ChromeDriver 90.0.4430.24)
- TensorFlow 2 (pip install tensorflow)

*If you want to scrape the dataset we used

