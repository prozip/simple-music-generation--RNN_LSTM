import numpy as np

class Convert:
    def __init__(self, songs):  # sort vocabulary list
        songs_joined = "\n\n".join(songs)
        vocab = sorted(set(songs_joined))
        print("There are", len(vocab), "unique chars in dataset")

        self.songs_joined = songs_joined
        self.vocab = vocab
        # vocabulary encode
        self.char2idx = {u: i for i, u in enumerate(vocab)}
        # vocabulary decode
        self.idx2char = np.array(vocab)

    def encode(self, str):  # Vectorize the songs string
        vectorized_output = np.array([self.char2idx[char] for char in str])
        return vectorized_output


class Preprocessing:
    def __init__(self, vectorized_songs):
        self.vectorized_songs = vectorized_songs

    def get_batch(self, seq_length, batch_size):
        # the length of the vectorized songs string
        n = self.vectorized_songs.shape[0] - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n-seq_length, batch_size)

        # list of input/output sequences for the training batch
        input_batch = [self.vectorized_songs[i: i+seq_length] for i in idx]
        output_batch = [self.vectorized_songs[i+1: i+seq_length+1]
                        for i in idx]

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch
