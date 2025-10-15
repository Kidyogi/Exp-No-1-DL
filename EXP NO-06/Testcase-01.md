code :

import torch
import torch.nn as nn
import torch.optim as optim

sentences = [["I", "love", "NLP"], ["He", "plays", "football"]]
tags = [["PRON", "VERB", "NOUN"], ["PRON", "VERB", "NOUN"]]

word2idx = {"<PAD>": 0}
tag2idx = {"<PAD>": 0}

for sent in sentences:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

for tag_seq in tags:
    for tag in tag_seq:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)

idx2tag = {i: t for t, i in tag2idx.items()}

def encode(seq, vocab):
    return [vocab[word] for word in seq]

X = [encode(s, word2idx) for s in sentences]
Y = [encode(t, tag2idx) for t in tags]

max_len = max(len(x) for x in X)
for i in range(len(X)):
    while len(X[i]) < max_len:
        X[i].append(word2idx["<PAD>"])
        Y[i].append(tag2idx["<PAD>"])

X = torch.tensor(X)
Y = torch.tensor(Y)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x) 
        outputs, hidden = self.rnn(emb)
        return outputs 
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs):
        outputs, _ = self.rnn(encoder_outputs)
        logits = self.fc(outputs)
        return logits

vocab_size = len(word2idx)
tagset_size = len(tag2idx)
embedding_dim = 32
hidden_size = 64

encoder = Encoder(vocab_size, embedding_dim, hidden_size)
decoder = Decoder(hidden_size, tagset_size)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

for epoch in range(200):
    encoder.train()
    decoder.train()
    
    optimizer.zero_grad()
    enc_outputs = encoder(X)
    logits = decoder(enc_outputs)

    loss = criterion(logits.view(-1, tagset_size), Y.view(-1))
    loss.backward()
    optimizer.step()

encoder.eval()
decoder.eval()
with torch.no_grad():
    enc_outputs = encoder(X)
    logits = decoder(enc_outputs)
    predictions = torch.argmax(logits, dim=2)

print("\nPOS Tagging using Seq2Seq Model")
print("Sentence\t\tPredicted Tags\t\tCorrect (Y/N)")

for i, sent in enumerate(sentences):
    pred_tags = [idx2tag[idx.item()] for idx in predictions[i][:len(sent)]]
    gold_tags = tags[i]
    correct = "Y" if pred_tags == gold_tags else "N"
    
    print(f"{' '.join(sent):<20} {' '.join(pred_tags):<20} {correct}")

output :
    
<img width="459" height="112" alt="Screenshot 2025-10-15 093855" src="https://github.com/user-attachments/assets/2981f9c1-e767-4eaf-8bea-a31cb1e16fbb" />
