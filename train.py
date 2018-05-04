import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from utils.data_loader import get_loader, Vocabulary
from utils.model import *
# from utils.logger import Logger


class Im2pGenerator(object):
    def __init__(self, _args):
        self.args = _args
        self.min_loss = 100000
        self.__init_model_path()
        self.transform = self.__init_transform()
        self.vocab = self.__init_vocab()
        self.train_data_loader = self.__init_data_loader(self.args.train_images_json)
        self.val_data_loader = self.__init_data_loader(self.args.val_images_json)

        self.encoderCNN = self.__init_encoderCNN()
        self.sentenceRNN = self.__init_sentenceRNN()
        self.wordRNN = self.__init_wordRNN()

        self.criterion = self.__init_criterion()
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.__init_scheduler()

        # self.logger = self.__init_logger()

    def train(self):
        for epoch in range(self.args.epochs):
            train_loss = self.__epoch_train()
            val_loss = self.__epoch_val()
            self.scheduler.step(val_loss)
            print("[{}] Epoch-{} - train loss:{} - val loss:{}".format(self.__get_now(),
                                                                       epoch + 1,
                                                                       train_loss,
                                                                       val_loss))
            self.__save_model(val_loss, self.args.saved_model_name)
            # self.__log(train_loss, val_loss, epoch + 1)

    def __epoch_train(self):
        train_loss = 0
        self.encoderCNN.train()
        self.wordRNN.train()
        self.sentenceRNN.train()

        for i, (images, _, captions, p_real) in enumerate(self.train_data_loader):
            images = self.__to_var(images, volatile=True)
            captions = self.__to_var(captions)
            pooled_vectors = self.encoderCNN.forward(images)

            sentence_states = None

            sentence_loss = 0
            word_loss = 0

            for sentence_index in range(captions.shape[1]):
                p, topic_vec, sentence_states = self.sentenceRNN.forward(pooled_vectors, sentence_states)
                sentence_loss += self.criterion(p, self.__to_var(p_real[:, sentence_index]))

                for word_index in range(1, captions.shape[2] - 1):
                    words_pred = self.wordRNN.forward(topic_vec=topic_vec,
                                                      captions=captions[:, sentence_index, :word_index])
                    words_real = torch.FloatTensor(captions.shape[0], len(self.vocab)).cuda().zero_()
                    words_real.scatter_(1, torch.unsqueeze(captions[:, sentence_index, word_index], 1).data, 1)
                    word_loss += self.criterion(words_pred, self.__to_var(words_real))
            loss = self.args.lambda_sentence * sentence_loss + self.args.lambda_word * word_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data[0]

        return train_loss

    def __epoch_val(self):
        val_loss = 0
        self.encoderCNN.eval()
        self.wordRNN.eval()
        self.sentenceRNN.eval()

        for i, (images, _, captions, p_real) in enumerate(self.train_data_loader):
            images = self.__to_var(images, volatile=True)
            captions = self.__to_var(captions)
            pooled_vectors = self.encoderCNN.forward(images)

            sentence_states = None

            sentence_loss = 0
            word_loss = 0

            for sentence_index in range(captions.shape[1]):
                p, topic_vec, sentence_states = self.sentenceRNN.forward(pooled_vectors, sentence_states)
                sentence_loss += self.criterion(p, self.__to_var(p_real[:, sentence_index]))

                for word_index in range(1, captions.shape[2] - 1):
                    words_pred = self.wordRNN.forward(topic_vec=topic_vec,
                                                      captions=captions[:, sentence_index, :word_index])
                    words_real = torch.FloatTensor(captions.shape[0], len(self.vocab)).cuda().zero_()
                    words_real.scatter_(1, torch.unsqueeze(captions[:, sentence_index, word_index], 1).data, 1)
                    word_loss += self.criterion(words_pred, self.__to_var(words_real))
            loss = self.args.lambda_sentence * sentence_loss + self.args.lambda_word * word_loss
            val_loss += loss.data[0]

        return val_loss

    def __init_data_loader(self, data_json):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 data_json=data_json,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=True)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_encoderCNN(self):
        decoder = EncoderCNN(P=self.args.P,
                             pretrained=self.args.pretrained)
        if self.args.cuda:
            decoder.cuda()
        return decoder

    def __init_sentenceRNN(self):
        sentenceRNN = SentenceRNN(hidden_size=self.args.hidden_size,
                                  lstm_layer=self.args.sentence_lstm_layer,
                                  pooling_dim=self.args.pooling_dim,
                                  topic_dim=self.args.topic_dim)
        if self.args.cuda:
            sentenceRNN.cuda()
        return sentenceRNN

    def __init_wordRNN(self):
        wordRNN = WordRNN(embed_size=self.args.embed_size,
                          hidden_size=self.args.hidden_size,
                          vocab_size=len(self.vocab),
                          num_layer=self.args.word_lstm_layer)
        if self.args.cuda:
            wordRNN.cuda()
        return wordRNN

    def __init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    # def __log(self, train_loss, val_loss, epoch):
    #     info = {
    #         'train loss': train_loss,
    #         'val loss': val_loss
    #     }
    #
    #     for tag, value in info.items():
    #         self.logger.scalar_summary(tag, value, epoch + 1)
    #
    # def __init_logger(self):
    #     logger = Logger(os.path.join(self.args.log_dir, self.__get_now()))
    #     return logger

    def __to_var(self, x, volatile=False):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def __get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def __get_now(self):
        return str(time.strftime('%y%m%d-%H:%M:%S', time.gmtime()))

    def __init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        return scheduler

    def __save_model(self, loss, file_name):
        if loss < self.min_loss:
            print("Saved Model in {}".format(file_name))
            torch.save({'encoderCNN': self.encoderCNN.state_dict(),
                        'sentenceRNN': self.sentenceRNN.state_dict(),
                        'wordRNN': self.wordRNN.state_dict(),
                        'best_loss': loss,
                        'optimizer': self.optimizer.state_dict()},
                       os.path.join(self.args.model_path, "{}.npz".format(file_name)))
            self.min_loss = loss

    def __init_criterion(self):
        return nn.BCELoss(size_average=False)

    def __init_optimizer(self):
        params = list(self.encoderCNN.parameters()) \
                 + list(self.wordRNN.parameters()) \
                 + list(self.sentenceRNN.parameters())
        return torch.optim.Adam(params=params, lr=self.args.learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_images_json', type=str, default='./data/train_split.json',
                        help='the train array')
    parser.add_argument('--val_images_json', type=str, default='./data/val_split.json',
                        help='the val array')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='the path for tensorboard')
    parser.add_argument('--saved_model_name', type=str, default='./val')

    # Model parameters
    parser.add_argument('--P', type=int, default=1024,
                        help='the dim for pooling vec')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='the size for embedding')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='the size for hidden states')
    parser.add_argument('--word_lstm_layer', type=int, default=2,
                        help='the num layer for word rnn')
    parser.add_argument('--sentence_lstm_layer', type=int, default=1,
                        help='the num layer for sentence rnn')
    parser.add_argument('--pooling_dim', type=int, default=1024,
                        help='the size for pooling vector')
    parser.add_argument('--topic_dim', type=int, default=1024,
                        help='the size for topic vector')
    parser.add_argument('--lambda_sentence', type=int, default=5,
                        help='the cost lambda for sentence loss function')
    parser.add_argument('--lambda_word', type=int, default=1,
                        help='the cost lambda for word loss function')

    parser.add_argument('--epochs', type=int, default=50,
                        help='the num of epochs when training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='the batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='the initial learning rate')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    model = Im2pGenerator(args)
    model.train()
