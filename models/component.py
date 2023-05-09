import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from transformers import BertModel
# from utils.data_utils import pad_tensor


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(in_features, out_features)))
        self.norm = nn.LayerNorm(out_features)
        if bias:
            self.bias = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(out_features,1)).squeeze())
        else:
            self.register_parameter('bias', None)
        self.relu = nn.ReLU()

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        # can remove + 1
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden.float()) / denom
        if self.bias is not None:
            return self.norm(self.relu(output + self.bias))
        else:
            return output


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len, h0=None):
        """
        from https://github.com/HITSZ-HLT/CMGCN/blob/main/layers/dynamic_rnn.py
        :param h0:
        :return:
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_len = torch.tensor(x_len)
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, _ = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, _ = self.RNN(x_emb_p, None)
            else:
                out_pack, _ = self.RNN(x_emb_p, h0)

        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
        out = out[0]  #
        out = out[x_unsort_idx]
        """unsort: out c"""
        if self.bidirectional:
            out = (out[:, :,: self.hidden_size] + out[:,:,self.hidden_size:])/2
        return out


class TextEncoder(nn.Module):
    r"""Initializes a NLP embedding block.
     :param input_size:
     :param nhead:
     :param dim_feedforward:
     :param dropout:
     :param activation:
     参数没有初始化
     """

    def __init__(self, input_size=768, out_size=300, rnn=False, rnn_type='LSTM', ch=False, finetune=True):
        super(TextEncoder, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.rnn = rnn
        self.rnn_type = rnn_type
        self.finetune = finetune
        self.ch = ch

        if self.rnn:
            self.lstm = DynamicLSTM(input_size=self.input_size, hidden_size=self.out_size, num_layers=1, bias=True,
                                    batch_first=True, dropout=0, bidirectional=False, rnn_type=self.rnn_type)
        else:
            self.lstm = None
        if self.ch:
            self.bert_model = BertModel.from_pretrained('//data/sunhao/bert/bert-base-chinese')
        else:
            self.bert_model = BertModel.from_pretrained('//data/sunhao/bert/bert-base-uncased')

        self.norm = nn.LayerNorm(self.out_size)
        self.linear1 = nn.Linear(self.input_size, 500)
        self.linear2 = nn.Linear(500, self.out_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, texts, word_seq, word_length):
        """
        Function to compute forward pass of the ImageEncoder TextEncoder
        Args:
            texts: (N,L,D) Padded Tensor. L is the length. D is dimension after bert
            token_length: (N) list of length
            word_seq:(N,tensor) list of tensor
            key_padding_mask: (N,L1) Tensor. L1 is the np length. True means mask

        Returns:
            t1: (N,L1,D). The embedding of each word or np. D is dimension after bert.
            score: (N,L1,D). The importance of each word or np. For convenience, expand the tensor (N,L1，D) to compute
            the caption embedding.


        """
        if self.finetune:
            texts = self.bert_model(**texts)[0]
        else:
            with torch.no_grad():
                texts = self.bert_model(**texts)[0]

        # remove cls token and sep token
        texts = texts[:, 1:-1, :]
        captions = []
        for i in range(texts.size(0)):
            # [X,L,H] X is the number of np and word
            captions.append(torch.stack([torch.mean(texts[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]]))
        texts = pad_sequence(captions, batch_first=True).cuda()
        if self.rnn:
            texts = self.lstm(texts, word_length)
        else:
            # two layer mlp
            texts = self.norm(self.relu2(self.linear2(self.relu1(self.linear1(texts)))))
        # (N,L,D)
        return texts


class ImageEncoder(nn.Module):
    def __init__(self, input_dim=768, inter_dim=500, output_dim=300):
        """
            Initializes the model to process bounding box features extracted by MaskRCNNExtractor
            Returns:
                None
        """
        super(ImageEncoder, self).__init__()
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, self.output_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm = torch.nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
            """
        x = self.relu2(self.fc2(self.relu1(self.fc1(x))))
        x = self.norm(x)
        return x


class MCO(nn.Module):
    def __init__(self, input_size=300, nhead=6, dim_feedforward=600, dropout=0.1):
        super(MCO, self).__init__()
        self.co_att = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        self.linear1 = nn.Linear(input_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MCO, self).__setstate__(state)

    def forward(self, tgt, src, src_key_padding_mask=None):
        """
        Args:
            tgt(L, N, E) : query matrix in MultiAttention. It's a Tensor.
            src(S, N, E) : Key, Value in MultiAttention. It's a Tensor.
            src_key_padding_mask(N,S) : if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored

        Shape for inputs:
            tgt(L, N, E)
        Returns:
            tgt(L,N,E): output of co-attention

        """
        if src_key_padding_mask is not None:
            # the tgt is image
            tgt2 = self.co_att(tgt, src, src, key_padding_mask=src_key_padding_mask)[0]
        else:
            # the tgt is text
            assert src_key_padding_mask is None, "The src has src_padding_mask but it's a image batch"
            tgt2 = self.co_att(tgt, src, src)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt
