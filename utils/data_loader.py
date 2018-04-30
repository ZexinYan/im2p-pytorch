import os
import pickle
import torch
import torch.utils.data as data
from PIL import Image
from utils.build_vocab import Vocabulary, JsonReader
import numpy as np
from torchvision import transforms


class DataSet(data.Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 data_json,
                 vocabulary,
                 transform=None):
        """

        :param image_dir: the path of image folder.
        :param caption_json: total caption file.
        :param data_json: split data json
        :param vocabulary: Vocabulary object
        :param transform: transforms
        """
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)
        self.data = JsonReader(data_json)
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, index):
        """

        :param index: id
        :return:
            image: (3, 224, 224)
            target: (sentence_num, word_num)
            image_id: 1
        """
        image_id = str(self.data[index])
        paragraph = self.caption[image_id]['paragraph']

        image = Image.open(os.path.join(self.image_dir, "{}.jpg".format(image_id))).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        target = list()
        word_num = 0
        for sentence in paragraph.split('. '):
            sentence = sentence.lower().replace('.', '').replace(',', '').split()
            if len(sentence) == 0:
                continue
            tokens = list()
            tokens.append(self.vocabulary('<start>'))
            tokens.extend([self.vocabulary(token) for token in sentence])
            tokens.append(self.vocabulary('<end>'))
            if word_num < len(tokens):
                word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        return image, image_id, target, sentence_num, word_num

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """
    :param data: list of tuple (image, target, image_id)
    :return:
        images: (batch_size, 3, 224, 224)
        targets: (batch_size, 6, 50)
        lengths: (batch_size, s_max)
        image_id: (batch_size, )
    """
    images, image_id, captions, sentences_num, words_num = zip(*data)
    images = torch.stack(images, 0)

    max_sentence_num = max(sentences_num)
    max_word_num = max(words_num)

    targets = np.zeros((len(captions), max_sentence_num, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    targets = torch.Tensor(targets).float()
    prob = torch.Tensor(prob)

    return images, image_id, targets, prob


def get_loader(image_dir,
               caption_json,
               data_json,
               vocabulary,
               transform,
               batch_size,
               shuffle):
    dataset = DataSet(image_dir=image_dir,
                      caption_json=caption_json,
                      data_json=data_json,
                      vocabulary=vocabulary,
                      transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':
    vocab_path = '../data/vocab.pkl'
    image_dir = '../data/images'
    caption_json = '../data/captions.json'
    data_json = '../data/train_split.json'
    batch_size = 3
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             caption_json=caption_json,
                             data_json=data_json,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=True)
    for i, (images, images_id, target, prob) in enumerate(data_loader):
        print("image:{}".format(images.shape))
        print("images id:{}".format(images_id))
        print("captions:{}".format(target))
        print("image id:{}".format(prob))
        break
