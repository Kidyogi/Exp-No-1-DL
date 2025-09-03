code :

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
data = "Deep learning is amazing. Deep learning builds intelligent systems."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequences = []
words = data.split()
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))
encoded = tokenizer.texts_to_sequences(sequences)
max_len = max([len(x) for x in encoded])
X = np.array([x[:-1] for x in pad_sequences(encoded, maxlen=max_len)])
y = to_categorical(
    [x[-1] for x in pad_sequences(encoded, maxlen=max_len)],
    num_classes=len(tokenizer.word_index) + 1
)
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_len - 1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)
def predict_next_word(model, tokenizer, text, max_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')
    pred_index = model.predict(sequence).argmax(axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            return word
    return ""
print("Next word predictions:")

output :

<img width="1161" height="341" alt="Screenshot 2025-09-03 100810" src="https://github.com/user-attachments/assets/df62c6a7-3a2a-40ae-8ed4-a44aefab2c17" />

for seq in sequences:
    next_word = predict_next_word(model, tokenizer, seq, max_len)
    print(f"['{seq}'] -> '{next_word}'")
