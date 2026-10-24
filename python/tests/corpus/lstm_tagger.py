# style: tutorial sequence tagger: embedding, bidirectional LSTM with an ignored state, log_softmax
# inputs: sentence=B,12:int64
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 16
HIDDEN_DIM = 32


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size=100, tagset_size=7, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores
