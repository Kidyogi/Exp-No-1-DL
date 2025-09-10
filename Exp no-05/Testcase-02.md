code :

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
data = {
    'Review': [
        "An emotional and deep plot",
        "The story was dull"
    ],
    'Expected': ['Positive', 'Negative']
}

df = pd.DataFrame(data)
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['Label'] = df['Expected'].map(label_map)
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])
padded = pad_sequences(sequences, padding='post', maxlen=10)
def build_model(cell_type='LSTM'):
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=16, input_length=10))
    if cell_type == 'LSTM':
        model.add(LSTM(32))
    else:
        model.add(GRU(32))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
lstm_model = build_model('LSTM')
lstm_model.fit(padded, df['Label'], epochs=10, verbose=0)
gru_model = build_model('GRU')
gru_model.fit(padded, df['Label'], epochs=10, verbose=0)
lstm_preds = np.argmax(lstm_model.predict(padded), axis=1)
gru_preds = np.argmax(gru_model.predict(padded), axis=1)
reverse_map = {v: k for k, v in label_map.items()}
df['LSTM Output'] = [reverse_map[i] for i in lstm_preds]
df['GRU Output'] = [reverse_map[i] for i in gru_preds]
df['Same?'] = df['LSTM Output'] == df['GRU Output']
df['Same?'] = df['Same?'].map({True: 'Yes', False: 'No'})
print(df[['Review', 'Expected', 'LSTM Output', 'GRU Output', 'Same?']])

output :

<img width="907" height="108" alt="Screenshot 2025-09-10 121504" src="https://github.com/user-attachments/assets/1c45cb9c-85a2-4872-80b7-122662dda394" />
