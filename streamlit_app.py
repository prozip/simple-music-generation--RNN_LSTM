from model import build_model
from datasets import Songs, extract_song_snippet, play_song
from utils import Convert, get_time
import tensorflow as tf
from tqdm import tqdm
import os, shutil
import streamlit as st


def generate_text(convert, model, start_string, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)
    '''TODO: convert the start string to numbers (vectorize)'''
    input_eval = [convert.char2idx[s] for s in start_string]  # TODO
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(convert.idx2char[predicted_id])  # TODO

    return (start_string + ''.join(text_generated))


def playback():
    try:
        os.mkdir('data/run')
    except:
        pass
    run_dir = 'data/run/' + get_time()
    os.mkdir(run_dir)
    generated_songs = extract_song_snippet(generated_text)

    for i, song in enumerate(generated_songs):
        st.markdown(song)
        f = open(run_dir + '/' + str(i) + '.abc','w')
        f.write(song)
        with open(run_dir + '/' + str(i) + '.abc', 'rb') as f:
            st.download_button('Download abc', f, file_name='music.abc')


def cover():
    st.title("Music Generaion with RNN/LSTM")
    st.write(os.getcwd())
    st.write(os.listdir())

if __name__ == "__main__":
    cover()
    # init
    songs = Songs()
    convert = Convert(songs.songs)

    # Model parameters:
    vocab_size = len(convert.vocab)
    embedding_dim = 256
    rnn_units = 1024  # Experiment between 1 and 2048

    checkpoint_dir = 'pretrain'
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    # Restore the model weights for the last checkpoint after training
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    print(model.summary())

    generated_text = generate_text(convert, model, "X", 1000)
    playback()

    
