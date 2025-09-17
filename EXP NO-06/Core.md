code :

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {word: i+1 for i, word in enumerate(word_vocab)}
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}

max_seq_len = max(len(sent.split()) for sent in input_texts)

encoder_input_data = [[word2idx[word] for word in sent.split()] for sent in input_texts]
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_seq_len, padding='post')

decoder_target_data = [[tag2idx[tag] for tag in tags] for tags in target_texts]
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_seq_len, padding='post')

decoder_input_data = np.zeros_like(decoder_target_data)
decoder_input_data[:, 1:] = decoder_target_data[:, :-1]

decoder_target_data = np.expand_dims(decoder_target_data, -1)

vocab_size = len(word_vocab) + 1  
tag_size = len(tag_vocab)

encoder_inputs = Input(shape=(max_seq_len,))
enc_emb = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_seq_len,))
dec_emb = Embedding(tag_size, 64, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(tag_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=2, epochs=50)


test_input = encoder_input_data[0:1]
decoder_input_test = np.zeros((1, max_seq_len))
decoder_output = model.predict([test_input, decoder_input_test])
predicted_tags = np.argmax(decoder_output, axis=-1)[0]

idx2tag = {v:k for k,v in tag2idx.items()}
predicted_tag_names = [idx2tag[idx] for idx in predicted_tags if idx != 0]

print("Input:", input_texts[0])
print("Predicted POS tags:", predicted_tag_names)

output :

<img width="304" height="35" alt="Screenshot 2025-09-17 100621" src="https://github.com/user-attachments/assets/c7a828d5-c9d1-4e36-89db-aa40e4c10f7b" />
