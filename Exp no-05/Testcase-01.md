code :

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
data = {
    'Review': [
        "I loved the movie, fantastic!",
        "Worst film ever, boring.",
        "It was okay, not great."
    ],
    'Sentiment': ['Positive', 'Negative', 'Neutral']
}
df = pd.DataFrame(data)
label_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
df['SentimentEncoded'] = df['Sentiment'].map(label_map)
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])
padded = pad_sequences(sequences, padding='post', maxlen=20)
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax') 
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(padded, df['SentimentEncoded'], epochs=10, verbose=1)
predictions = model.predict(padded)
predicted_labels = np.argmax(predictions, axis=1)
reverse_label_map = {v: k for k, v in label_map.items()}
df['PredictedSentiment'] = [reverse_label_map[label] for label in predicted_labels]
df['Correct'] = df['SentimentEncoded'] == predicted_labels
df['Correct'] = df['Correct'].map({True: 'Y', False: 'N'})
print(df[['Review', 'Sentiment', 'PredictedSentiment', 'Correct']])

output :

<img width="600" height="584" alt="Screenshot 2025-09-10 120251" src="https://github.com/user-attachments/assets/0aaa585c-5f9c-42b0-9f10-ebc8f2fe990f" />
