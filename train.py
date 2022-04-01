import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import mitdeeplearning as mdl

from datasets import Songs
from utils import Convert, Preprocessing
from model import build_model

def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

@tf.function
def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)  
        loss = compute_loss(y, y_hat)  # TODO

    # Now, compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)  # TODO

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


if __name__ == "__main__":
    # init
    songs = Songs()
    convert = Convert(songs.songs)
    vectorized_songs = convert.encode(convert.songs_joined)
    preprocessing = Preprocessing(vectorized_songs)

    # Optimization parameters:
    num_training_iterations = 200  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    seq_length = 100  # Experiment between 50 and 500
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

    # Model parameters:
    vocab_size = len(convert.vocab)
    embedding_dim = 256
    rnn_units = 1024  # Experiment between 1 and 2048

    # Checkpoint location:
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    # Build model
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Begin training!
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for iter in tqdm(range(num_training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = preprocessing.get_batch(seq_length, batch_size)
        loss = train_step(x_batch, y_batch)
        print(" loss =", loss.numpy().mean())

        # Update the model with the changed weights!
        if iter % 100 == 0:     
            model.save_weights(checkpoint_prefix)
            
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)
