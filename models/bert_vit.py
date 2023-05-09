import torch.nn as nn
from models.component import TextEncoder, ImageEncoder
import torch


class Bert_Vit(nn.Module):
    def __init__(self, input_size=768, out_size=300, rnn=False, rnn_type='LSTM', ch=False, finetune=True, type='vit'):
        super(Bert_Vit, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.rnn = rnn
        self.rnn_type = rnn_type
        self.ch = ch
        self.finetune = finetune
        self.type = type

        # our framework
        self.text_encoder = TextEncoder(input_size=self.input_size, out_size=self.out_size, rnn=self.rnn,
                                        rnn_type=self.rnn_type, ch=self.ch, finetune=self.finetune)
        self.img_encoder = ImageEncoder(input_dim=512, inter_dim=500, output_dim=self.out_size)

        self.patch_score = nn.Linear(self.out_size, 1)
        self.token_score = nn.Linear(self.out_size, 1)

        if self.type == 'vit':
            self.classifier = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size),
                                            nn.ReLU(),
                                            nn.Linear(in_features=self.out_size, out_features=2),
                                            nn.ReLU())
        elif self.type == 'bert':
            self.classifier = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size),
                                            nn.ReLU(),
                                            nn.Linear(in_features=self.out_size, out_features=2),
                                            nn.ReLU())
        elif self.type == 'vitbert':
            self.classifier = nn.Sequential(nn.Linear(in_features=2*self.out_size, out_features=self.out_size),
                                            nn.ReLU(),
                                            nn.Linear(in_features=self.out_size, out_features=2),
                                            nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, imgs, encoded_texts, word_len, word_spans):
        """generate instance for each predicate"""
        # (N,r,outdim)
        imgs_0 = self.img_encoder(imgs)
        # (N, L, outdim)
        texts_0 = self.text_encoder(encoded_texts, word_spans, word_len)

        I_score = self.softmax(self.patch_score(imgs_0).squeeze())
        T_score = self.softmax(self.token_score(texts_0).squeeze())
        # N, D
        I_emb = torch.bmm(imgs_0.permute(0, 2, 1), I_score.unsqueeze(2)).squeeze()
        T_emb = torch.bmm(texts_0.permute(0, 2, 1), T_score.unsqueeze(2)).squeeze()

        if self.type == 'vit':
            final_score = self.classifier(T_emb)
        elif self.type == 'bert':
            final_score = self.classifier(T_emb)
        else:
            con = torch.cat([T_emb, I_emb], dim=-1)
            final_score = self.classifier(con)
        return final_score



