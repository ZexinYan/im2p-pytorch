import pickle
import argparse
from collections import Counter
import json


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<end>')
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __getitem__(self, item):
        if item > len(self.id2word):
            return '<unk>'
        return self.id2word[item]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    caption_reader = JsonReader(json_file)
    counter = Counter()

    for items in caption_reader:
        paragraph = items['paragraph']
        paragraph = paragraph.replace(',', ' ').replace('.', '').replace('\"', '')
        counter.update(paragraph.lower().split(' '))
    words = [word for word, cnt in counter.items() if cnt >= threshold and word != '']
    vocab = Vocabulary()

    for _, word in enumerate(words):
        if len(word) > 1:
            vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json_file=args.json_file,
                        threshold=args.threshold)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total Vocabulary Size:{}".format(len(vocab)))
    print("Saved Path in {}".format(args.vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        default='../data/paragraphs_v1.json',
                        help='path for caption file')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=10,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
