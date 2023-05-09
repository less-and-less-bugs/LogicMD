import torch
from torch.utils.data import Dataset
import json
"""
--------------------------------dataset pre-process----------------------------------------
This file aims to the class of dataloader and dataset of sarcasm, twitter, weibo dataset.
sarcasm is pre-processed by data_process.ipython.
twitter dataset is pre-processed by twitter_sarcaweibo_processsm_process.ipython.
weibo dataset is pre-processed by weibo_process.ipython.

dataset format:
twitter:
[twitter_id, text, img_id('sandyA_fake_29.jpg'), label, caption, dependency of text, dependency of caption]
 {'chunk_cap': ['scary shit', '#', 'hurricane', '#', 'ny'],
  'token_cap': ['scary', 'shit', '#', 'hurricane', '#', 'ny'],
  'token_dep': [[0, 1], [3, 1], [5, 3]],
  'chunk_dep': [[2, 0], [4, 2]],
  'chunk': ['scary shit', 'hurricane', 'ny'],
  'chunk_index': [0, 2, 4]},

Weibo:
[img_id, text, label, event_label, dependency of text, dependency of caption]
{'token_cap':, 'token_dep':}
"""
# '/data/sunhao/rulenews/dataset/twitter/embedding.pt'
# '/data/sunhao/rulenews/dataset/twitter/embedding34.pt'


class Twitter_Set(Dataset):
    def __init__(self, text_path='/data/sunhao/rulenews/dataset/twitter/texts/twitter_final2.json',
                 img_path='/data/sunhao/rulenews/dataset/twitter/embedding34.pt', max_length=150, mode='random', phase='train'):
        self.text_path = text_path
        self.img_path = img_path
        self.max_length = max_length
        self.mode = mode
        self.phase = phase
        if self.mode == 'random':
            with open(self.text_path, "r", encoding='utf8') as f:
                self.data = json.load(f)
        else:
            with open(self.text_path, "r", encoding='utf8') as f:
                data = json.load(f)
            if self.phase == 'train':
                self.data = data[:8617]
            else:
                self.data = data[8617:]

        # with open(img_path, 'r')
        self.img_set = torch.load(self.img_path)

    def __getitem__(self, index):
        data = self.data[index]
        text = data[5]['token_cap']
        text_dep = data[5]['token_dep']
        img_id = data[2]
        # (49, 768)
        img = self.img_set[img_id]
        label = int(data[3])
        return img, text, text_dep, label

    def __len__(self):
        return len(self.data)


class Weibo_Set(Dataset):
    def __init__(self, text_path="//data/sunhao/rulenews/dataset/weibo/texts/whole_set.json",
                 img_path="//data/sunhao/rulenews/dataset/weibo/embedding34.pt", max_length=150, mode='random', phase='train'):
        # "//data/sunhao/rulenews/dataset/weibo/embedding.pt" for vit
        self.text_path = text_path
        self.img_path = img_path
        self.max_length = max_length
        self.mode = mode
        self.phase = phase

        if self.mode == 'random':
            with open(self.text_path, "r", encoding='utf8') as f:
                self.data = json.load(f)
        else:
            if self.phase == 'train':
                with open("//data/sunhao/rulenews/dataset/weibo/texts/train_dep_set.json", "r", encoding='utf8') as f:
                    self.data = json.load(f)
            else:
                with open("//data/sunhao/rulenews/dataset/weibo/texts/test_dep_set.json", "r", encoding='utf8') as f:
                    self.data = json.load(f)
        # with open(img_path, 'r')
        self.img_set = torch.load(self.img_path)

    def __getitem__(self, index):
        data = self.data[index]
        text = data[4]['token_cap']
        text_dep = data[4]['token_dep']
        img_id = data[0]
        # (49, 768)
        img = self.img_set[img_id]
        label = int(data[3])
        return img, text, text_dep, label

    def __len__(self):
        return len(self.data)


class Sarcasm_Set(Dataset):
    def __init__(self, type="train", max_length=150, text_path=None, img_path=None):
        """
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding. Resnet152(,2048), Vit B_32(,768), Vit L_32(, 1024)
            use_np: True or False, whether use noun phrase as relation matching node. It is useless in this paper.
            img_path:
            knowledge: 1 caption, 2 ANP, 3 attribute, 0 not use knowledge
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.img_set = torch.load(self.img_path)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            img: (49, 768). Tensor.
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        if self.type == "train":
            label = sample[2]
            text_ = sample[3]
        else:
            # label =sample[2] hashtag label
            label = sample[3]
            text_ = sample[4]
        # useless in this project

        text = text_["token_cap"]
        text_dep = text_["token_dep"]

        img = self.img_set[index]

        # caption
        return img, text, text_dep, label

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)





