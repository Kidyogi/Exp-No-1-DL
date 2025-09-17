code :

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
max_len = 10 
n_words = 5000 
n_tags = 5 
tag_values = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
model = Bidirectional(LSTM(units=50, return_sequences=True))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
model = Model(input, out)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
words = ["John", "lives", "in", "New", "York"]
X_example = np.array([[1, 2, 3, 4, 5] + [0]*(max_len-len(words))])
y_pred = model.predict(X_example)
y_pred_classes = np.argmax(y_pred, axis=-1)[0]
result = []
for word, tag_idx in zip(words, y_pred_classes[:len(words)]):
    result.append([word, tag_values[tag_idx]])
df = pd.DataFrame(result, columns=["Word", "BIO Tag"])
print(df)

output :

<img width="278" height="113" alt="Screenshot 2025-09-17 104356" src="https://github.com/user-attachments/assets/2d0bc827-2b43-4f0e-b8ed-f4c3673c42a1" />
