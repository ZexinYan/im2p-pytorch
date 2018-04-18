import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class RegionPoolingModel(nn.Module):
    def __init__(self, feature_dim=4096, pooling_dim=1024):
        super(RegionPoolingModel, self).__init__()
        self.feature_dim = feature_dim
        self.pooling_dim = pooling_dim
        self.linear = nn.Linear(feature_dim, pooling_dim)
        self.__init_weights()

    def __init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, region_features):
        pooling_vector = self.linear(region_features)
        return torch.max(pooling_vector, 1)[1]


class SentenceRNN(nn.Module):
    def __init__(self, hidden_size=512, lstm_layer=1, pooling_dim=1024, topic_dim=1024, s_max=6, batch_size=10):
        super(SentenceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.pooling_dim = pooling_dim
        self.topic_dim = topic_dim
        self.S_max = s_max
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.topic_dim, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=True)
        self.logistic = nn.Linear(self.hidden_size, 1)
        self.logistic = self.__init_weights(self.logistic)
        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(self.hidden_size, self.topic_dim)
        self.fc1 = self.__init_weights(self.fc1)
        self.fc2 = nn.Linear(self.topic_dim, self.topic_dim)
        self.fc2 = self.__init_weights(self.fc2)

    def __init_weights(self, func):
        func.weight.data.uniform_(-0.1, 0.1)
        func.bias.data.fill_(0)
        return func

    def forward(self, pooling_vector):
        pooling_vector = pooling_vector.unsqueeze(1)
        p_vec = Variable(torch.zeros((self.batch_size, self.S_max)))
        topic_vec = Variable(torch.zeros(self.batch_size, self.S_max, self.pooling_dim))

        states = None
        for i in range(self.S_max):
            hiddens, states = self.lstm(pooling_vector, states)
            p_i = self.activation(self.logistic(hiddens))
            topic_i = self.fc2(self.fc1(hiddens))
            p_vec[:, i] = p_i.squeeze().data
            topic_vec[:, i] = topic_i
        return p_vec, topic_vec

    def sample(self, pooling_vector):
        # TODO: Sample method
        pass


class WordRNN(nn.Module):
    def __init__(self, embed_size=1024, hidden_size=512, vocab_size=10000, num_layer=2, s_max=6, n_max=50, batch_size=10):
        super(WordRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.s_max = s_max
        self.batch_size = batch_size
        self.n_max = n_max

        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions, lengths):
        """

        :param topic_vec: [batch_size, s_max, pooling_dim]
        :param captions: [batch_size, s_max, n_max]
        :param lengths: [batch_size, s_max]
        :return:
        """
        topic_vec = topic_vec.view(-1, self.embed_size).unsqueeze(1)
        captions = captions.view(-1, self.n_max)
        lengths = list(np.reshape(lengths, (-1,)))
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        print(embeddings.shape)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        print(hidden)
        outputs = self.linear(hidden[0])
        return outputs

    def sample(self):
        # TODO: Sample Method
        pass

if __name__ == '__main__':
    region_pooling_model = RegionPoolingModel()
    sentence_model = SentenceRNN()
    word_model = WordRNN()

    region_features = torch.randn(10, 50, 4096)
    region_features = Variable(region_features)
    result = region_pooling_model.forward(region_features).float()
    p_vec, topic_vec = sentence_model.forward(result)

    caption = Variable(torch.ones(10, 6, 50)).long()
    lengths = np.ones((10, 6), dtype=int) + 1
    outputs = word_model.forward(topic_vec, caption, lengths)
    print(outputs.shape)
