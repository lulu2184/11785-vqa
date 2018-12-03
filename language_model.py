import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, embedding_dim, np_file):
        super(WordEmbedding, self).__init__()

        self.embedding = nn.Embedding(ntoken + 1,
                                      embedding_dim,
                                      padding_idx=ntoken)
        self.ntoken = ntoken
        self.embedding_dim = embedding_dim

        weight_init = torch.from_numpy(np.load(np_file))
        self.embedding.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        x = self.embedding(x)
        return x


class QuestionEmbedding(nn.Module):
    def __init__(self, features_dim, hidden_num, nlayers, bidirectional=False):
        super(QuestionEmbedding, self).__init__()

        self.rnn = nn.GRU(features_dim, hidden_num, nlayers,
                          bidirectional=bidirectional, batch_first=True)
        self.hidden_num = hidden_num
        self.ndirection = 1 + int(bidirectional)

    def forward(self, x):
        # x: [batch, sequence, features_dim]
        batch_size = x.size(0)
        weight = next(self.parameters()).data
        hidden_shape = (
            self.nlayers * self.ndirections, batch_size, self.hidden_num)
        hidden_state = Variable(
            weight.new(self.nlayers, hidden_shape, self.nhid).zero_())
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
