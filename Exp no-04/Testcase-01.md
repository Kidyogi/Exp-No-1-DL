code :

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
sentences = [
    "Deep learning is amazing",
    "Deep learning builds intelligent",
    "Intelligent systems can learn"
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=total_words)
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(SimpleRNN(50))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
test_inputs = [
    "Deep learning is",
    "Deep learning builds",
    "Intelligent systems can learn"
]
for test_input in test_inputs:
    next_word = predict_next_word(model, tokenizer, test_input, max_seq_len)
    print(f"Input Text: '{test_input}' -> Predicted Next Word: '{next_word}'")

output :

<img width="543" height="57" alt="Screenshot 2025-09-03 110327" src="https://github.com/user-attachments/assets/abc5ebec-91cc-4451-936d-c4bfa6d988d2" />

