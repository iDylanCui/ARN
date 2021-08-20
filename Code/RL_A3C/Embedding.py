import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_size, padding, word_dropout_rate, isTrainEmbedding = True):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size

        self.embedding = nn.Embedding(self.vocab_size, self.input_size, padding_idx = padding)
        self.EDropout = nn.Dropout(word_dropout_rate)

        if isTrainEmbedding == False:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.EDropout(embeddings)
        return embeddings