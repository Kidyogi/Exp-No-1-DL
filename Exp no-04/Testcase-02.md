code :

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample Shakespeare-like text for training
text = """
To be or not to be that is the question
What light through yonder window breaks
"""

# Preprocess the text
text = text.lower()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create sequences of words for training
input_sequences = []
token_list = tokenizer.texts_to_sequences([text])[0]

for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_sequences.append(n_gram_seq)

# Pad sequences
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

input_sequences = np.array(input_sequences)

# Split into features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len - 1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_seq_len):
    sequence = tokenizer.texts_to_sequences([text.lower()])[0]
    sequence = pad_sequences([sequence], maxlen=max_seq_len - 1, padding='pre')
    pred = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(pred, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Test inputs from your snapshot
test_inputs = [
    "To be or not",
    "What light through yonder window"
]

for test_input in test_inputs:
    next_word = predict_next_word(model, tokenizer, test_input, max_seq_len)
    print(f"Input Sequence: '{test_input}' -> Predicted Word: '{next_word}'")

output :

<img width="623" height="58" alt="image" src="https://github.com/user-attachments/assets/f4582afc-b6c1-486e-9294-601b53aff044" />
