import argparse
import json
import os
import pickle

import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from utils.model import *
from utils.data_loader import *
from utils.build_vocab import *


class Sampler(object):
    def __init__(self, args):
        self.args = args
        self.vocabulary = self.__init_vocab()
        self.transform = self.__init_transform()
        self.data_loader = self.__load_data_loader()
        self.encoderCNN = self.__load_encoderCNN()
        self.wordRNN = self.__load_wordRNN()
        self.sentenceRNN = self.__load_sentenceRNN()
        self.__init_result_path()

    def sample(self):
        progress_bar = tqdm(self.data_loader, desc='Sampling')
        results = {}
        for images, images_id, targets, _ in progress_bar:
            for id in images_id:
                results[id] = {
                    'GT': '',
                    'Pred': ''
                }

            images = self.__to_var(images)

            pooled_vec = self.encoderCNN.forward(images)
            sentence_states = None
            for i in range(self.args.s_max):
                p, topic, sentence_states = self.sentenceRNN.sample(pooled_vec, sentence_states)
                samples_ids = self.wordRNN.sample(topic, self.args.n_max)
                p = (p > 0.4).squeeze(1)
                samples_ids = samples_ids * p

                for pred, reference, id in zip(samples_ids, targets[:, i, :], images_id):
                    results[id] = {
                        'GT': results[id]['GT'] + self.__vec2sent(reference),
                        'Pred': results[id]['Pred'] + self.__vec2sent(pred)
                    }

        self.__save_json(results)

    def __save_json(self, result):
        with open(os.path.join(self.args.result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocabulary[int(word_id)]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption) + '. '

    def __init_result_path(self):
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __load_encoderCNN(self):
        decoder = EncoderCNN(P=self.args.P,
                             pretrained=False)
        decoder.load_state_dict(torch.load(self.args.model_path)['encoderCNN'])

        if self.args.cuda:
            decoder.cuda()

        decoder.eval()
        return decoder

    def __load_sentenceRNN(self):
        sentenceRNN = SentenceRNN(hidden_size=self.args.hidden_size,
                                  lstm_layer=self.args.sentence_lstm_layer,
                                  pooling_dim=self.args.pooling_dim,
                                  topic_dim=self.args.topic_dim)
        sentenceRNN.load_state_dict(torch.load(self.args.model_path)['sentenceRNN'])

        if self.args.cuda:
            sentenceRNN.cuda()

        sentenceRNN.eval()
        return sentenceRNN

    def __load_wordRNN(self):
        wordRNN = WordRNN(embed_size=self.args.embed_size,
                          hidden_size=self.args.hidden_size,
                          vocab_size=len(self.vocabulary),
                          num_layer=self.args.word_lstm_layer)
        wordRNN.load_state_dict(torch.load(self.args.model_path)['wordRNN'])

        if self.args.cuda:
            wordRNN.cuda()

        wordRNN.eval()
        return wordRNN

    def __load_data_loader(self):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 data_json=self.args.data_json,
                                 vocabulary=self.vocabulary,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=False)
        return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model_path', type=str, default='./models/val.npz',
                        help='path for saving trained models')
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--data_json', type=str, default='./data/test_split.json',
                        help='the sample file array')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='the path for tensorboard')
    parser.add_argument('--result_name', type=str, default='test',
                        help='the result saved file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='the batch size for training')
    parser.add_argument('--result_path', type=str, default='./results',
                        help='the path for storing results')

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
    parser.add_argument('--s_max', type=int, default=6,
                        help='the max num of sentences')
    parser.add_argument('--n_max', type=int, default=50,
                        help='the max num of words for each sentence.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    sampler = Sampler(args)
    sampler.sample()
