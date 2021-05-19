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

Three different MM based models were built and tried out 
1. Markov model
2. N-gram model
3. N-gram model with Laplace smoothing and backoff

**Data**: 50, 100 or 200 musical pieces.

**Tokens**: `wait:0.25`; `77:0.25`. Left hand side represents the note and right hand side represents the duration.

### Markov model generator

Markov models are probabilistic models that assume that we can predict the probability of some future event by not looking too far into the past. 

In our case `P(next note | previous note)`.

In a sequence of notes, `wait:0.25 77:0.25 wait:0.25 70:0.25 74:0.25 X`, it means that `P(X|74:0.25)`.

As anticipated, this model did not provide good results- very similar to random sequence of notes.

###  N-gram model

In general, it is a Markov model that takes account not only 1 but N past events in a sequence.

In a sequence of notes: `wait:0.25 77:0.25 wait:0.25 70:0.25 74:0.25 57:0.5 X`, 4-gram means that `P(X|70:0.25 74:0.25 57:0.5)`.

The transition probability matrix grew fast and become very sparse. That problem was dealt robustly in case of this model- sample as long as it results in an already existing n-gram. 

This model proved to work best out of the three MM based models. The results were not random anymore but as the N grew, the results started to resemble the tracks from the corpus more and more.

### Advanced N-gram model

Everything is same as in N-gram model except the solution to the sparsity problem.

The sparsity problem dealt with backoff and laplace smoothing known from the NLP theory.
Backoff: 
```
while sampled n-gram not in current dictionary:
	look for the n-1-gram from the one lower dictionary
  ```
It will always succeed as the bi-gram dictionary holds all the possible transition probabilities from one note to other. 

The result is quite bad as the backoff throws in a lot of randomness. Although, the results can be considered somewhat original now. 

### Further improvements

Initially it was planned to try out hidden Markov model based generation as well, but unfortunately due to the time constraint, was not implemented.
Also, more musical theory could be taken into account.
## 4. Evaluation

We decided to evaluate the generator model(s) we created in both objective and subjective manner.

### Objective evaluation

For the objective evaluation we trained a model to classify the tracks as classical or nonclassical. We combined the dataset to train and test the model from the classical tracks dataset (that were also used for our LSTM and Markov models) and as the "nonclassical" tracks we generated 500 pieces using our Markov model. To ensure that the distribution of the MIDI length of the Markov model generated tracks was the same as the one of the classical tracks, we visually fitted a Log Normal Distribution with parameters mu=8.5 and sigma=sqrt(0.5). We tested the goodness of fit with the Cramer-Von Mises test and using the 95% confidence level we couldn't prove that the classical tracks weren't from said distribution. The MIDIs were then generated with lengths sampled from that distribution.

Only tracks shorter than 10 000 MIDI elements were incorporated in the final dataset for the evaluation model, which meant that the final dataset consisted of 541 tracks (310 Markov generated and 231 classical tracks). The preprocessing steps of tokenization and building of the dictionary were done in a similar manner as for the LSTM model input.

The evaluation model is a simple GRU model with one dropout layer (0.5) after the GRU layer and the model output is calculated using the sigmoid activation function. We opted for a relatively simple model due to the small dataset size.

The model was trained with batch size 64 for 12 epochs and achieved accuracy of 98.15% and recall of 96.67%. We then generated 50 tracks with our LSTM generator model and predicted whether the track was classical using the model. The mean posterior probability of class 1(classical) was 0.62 and the median was 0.82. In total 32/50=64% of the tracks were classified as classical music.

### Subjective evaluation

We also descided to evaluate the Markov and LSTM generated tracks using a test group. We put together a ten-part questionnaire with 3 tracks in each part: Markov- and LSTM-generated tracks and one classical piece from our original training set. The participants had to rank tracks in each part based on how computer-produced they felt each given track sounded (1. position for most authenticly classical track and 3. position for most generated sounding track). The tracks were randomly shuffled for each part.

From the overall results we saw that unsurprisingly, the most assigned position for all classical tracks was the first, for LSTM-generated the second and for Markov-generated the third. When looking into the results in the scope of parts, we saw again that the classical track was ranked first with a overwhelming amount of votes in almost all of the parts. The part 9 stood out in that way as an oddity, because in that part, the LSTM-generated track was the most popular choice for the first position.

In conclusion it seems that the LSTM- and Markov-generated tracks didn't sound authentic enough for the listeners and so they managed to identify the authentic classical piece from the three tracks with ease for the most part. When comparing the two methods of generation, the Markov-generated tracks were found to be less authentic by the listeners when compared to LSTM tracks.

## 5. Requirements

- Jupyter (or whatever you use to handle .ipynb files)
- Music21 (pip install music21)
- Selenium* (pip install selenium, with ChromeDriver 90.0.4430.24)
- TensorFlow 2 (pip install tensorflow)

*If you want to scrape the dataset we used

