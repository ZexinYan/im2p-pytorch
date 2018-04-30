import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, P=1024, pretrained=True):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, P)
        self.bn = nn.BatchNorm1d(P, momentum=0.01)
        self.__init_weights()

    def __init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


# class RegionPoolingModel(nn.Module):
#     def __init__(self, feature_dim=4096, pooling_dim=1024):
#         super(RegionPoolingModel, self).__init__()
#         self.feature_dim = feature_dim
#         self.pooling_dim = pooling_dim
#         self.linear = nn.Linear(feature_dim, pooling_dim)
#         self.__init_weights()
#
#     def __init_weights(self):
#         self.linear.weight.data.normal_(0, 0.01)
#         self.linear.bias.data.fill_(0)
#
#     def forward(self, region_features):
#         pooling_vector = self.linear(region_features)
#         return torch.max(pooling_vector, 1)[1]


class SentenceRNN(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 lstm_layer=1,
                 pooling_dim=1024,
                 topic_dim=1024):
        super(SentenceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.pooling_dim = pooling_dim
        self.topic_dim = topic_dim

        self.lstm = nn.LSTM(self.topic_dim, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=True)
        self.logistic = nn.Linear(self.hidden_size, 1)
        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(self.hidden_size, self.topic_dim)
        self.fc2 = nn.Linear(self.topic_dim, self.topic_dim)
        self.__init_weights()

    def __init_weights(self):
        self.logistic.weight.data.uniform_(-0.1, 0.1)
        self.logistic.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)

    def forward(self, pooling_vector, states=None):
        pooling_vector = pooling_vector.unsqueeze(1)

        hiddens, states = self.lstm(pooling_vector, states)
        p = self.activation(self.logistic(hiddens))
        topic = self.fc2(self.fc1(hiddens))
        return p, topic, states

    def sample(self, pooling_vector, states=None):
        pooling_vector = pooling_vector.unsqueeze(1)

        hiddens, states = self.lstm(pooling_vector, states)
        p = self.activation(self.logistic(hiddens))
        topic = self.fc2(self.fc1(hiddens))
        return p, topic, states


class WordRNN(nn.Module):
    def __init__(self,
                 embed_size=1024,
                 hidden_size=512,
                 vocab_size=10000,
                 num_layer=2):
        super(WordRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions):
        """
        :param topic_vec: [batch_size, pooling_dim]
        :param captions: [batch_size, n_max]
        :param lengths: [batch_size]
        :return:
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = nn.Softmax()(self.linear(hidden))
        return outputs[:, -1]

    def sample(self, features, n_max, states=None):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        inputs = features.unsqueeze(1)
        for i in range(n_max):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids


if __name__ == '__main__':
    decoder = EncoderCNN(pretrained=False)
    sentence_model = SentenceRNN()
    word_model = WordRNN()
    s_max = 6
    n_max = 50

    images = Variable(torch.randn(10, 3, 224, 224))
    region_features = decoder.forward(images)
    states = None

    for i in range(s_max):
        p, topic, states = sentence_model.forward(region_features, states)
        print(p)
        print(topic)
        print(states)
        break

