code :

import numpy as np
num_samples = 1000
max_encoder_seq_length = 15
max_decoder_seq_length = 15
num_encoder_tokens = 1000
num_decoder_tokens = 1000
encoder_input_data = np.random.randint(1, num_encoder_tokens, size=(num_samples, max_encoder_seq_length))
decoder_input_data = np.random.randint(1, num_decoder_tokens, size=(num_samples, max_decoder_seq_length))
decoder_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
for i, seq in enumerate(decoder_input_data):
    for t, word_id in enumerate(seq):
        if word_id > 0:
            decoder_target_data[i, t, word_id] = 1.0
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

output :

<img width="511" height="583" alt="Screenshot 2025-10-15 101423" src="https://github.com/user-attachments/assets/995c069e-6717-4f25-b618-727e64f636c5" />
