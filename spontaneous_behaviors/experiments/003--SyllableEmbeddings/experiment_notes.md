https://chatgpt.com/g/g-p-67b49a58bff88191838dd62962204448-neuraldatascience-mitneuralcomputation/c/69da2d4e-b764-838f-a284-0e41c7de7928

We should train some sort of a neural network with an embedding layer for syllables.
One possibility is predicting neural activity. 

This  is because the syllable timeseries is meaningless in terms of numbers. It's a class label and it should have embeddings.

Prediction at population level is noisy and not good. Attempting to predict PC's.

The idea would be to use these embeddings of syllables as an auxiliary variable. If we supervise
with neural data, we will get data leakage. Perhaps it would be better to have a classification
loss for predicting which syllable is next? That way we could use these embeddings in standalone analysis.

Current model:
A few important nuances:

It is not predicting raw neural activity directly. It predicts the PCA-compressed neural representation.
It is one-step Markov-style prediction: only current x_t and current syll_t are used, not a longer history.

For out of sample evaluation: one-step-ahead out-of-sample prediction with teacher forcing