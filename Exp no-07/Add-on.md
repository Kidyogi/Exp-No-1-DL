code :

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

vocab_inp_size = 5000
vocab_tar_size = 5000
embedding_dim = 256
units = 512
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 2)
        score = self.V(tf.nn.tanh(self.W1(values)[:, tf.newaxis, :, :] + self.W2(query_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=2)

        context_vector = attention_weights * values[:, tf.newaxis, :, :]
        context_vector = tf.reduce_sum(context_vector, axis=2)
        return context_vector, attention_weights
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):

        x = self.embedding(x)  
        if len(hidden.shape) == 2:
            dec_seq_len = x.shape[1]
            hidden = tf.repeat(tf.expand_dims(hidden, 1), repeats=dec_seq_len, axis=1)

        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([context_vector, x], axis=-1)

        output, state_h, state_c = self.lstm(x)
        output = self.fc(output) 

        return output, state_h, state_c, attention_weights
batch_size = 1
input_seq_len = 10  
decoder_seq_len = 5 
encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units)
encoder_input = tf.random.uniform((batch_size, input_seq_len), maxval=vocab_inp_size, dtype=tf.int32)
enc_output, enc_hidden_h, enc_hidden_c = encoder(encoder_input)
print("Encoder output shape:", enc_output.shape)
decoder_input = tf.random.uniform((batch_size, decoder_seq_len), maxval=vocab_tar_size, dtype=tf.int32)
dec_hidden = enc_hidden_h
dec_output, dec_hidden_h, dec_hidden_c, attention_weights = decoder(decoder_input, dec_hidden, enc_output)
print("Decoder output shape:", dec_output.shape)  
print("Attention weights shape:", attention_weights.shape) 

output :

<img width="313" height="55" alt="Screenshot 2025-10-15 112044" src="https://github.com/user-attachments/assets/ed004647-2f4c-47c2-aec0-30140db8fbbc" />
