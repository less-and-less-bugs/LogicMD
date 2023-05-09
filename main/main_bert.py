

import os
import json
import argparse
import time
from typing import List, Dict, Any
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
import nltk

import sys
import _pickle as pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import numpy as np
from random import randint
import random
import spacy
import sent2vec
from transformers import BertTokenizer, BertModel
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import LongformerModel, LongformerTokenizer
import regex as re
from sparsemax import Sparsemax
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained("bert-base-uncased", \
                                        output_hidden_states=False,
                                        output_attentions=False).cuda()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# tokenizer.add_tokens(['__ent1__', '__ent2__'])
# tokenizer.add_tokens(['[e1]', '[/e1]', '[e2]', '[/e2]'])
# bert_model.resize_token_embeddings(len(tokenizer))

# bert_model = RobertaModel.from_pretrained('roberta-base').cuda()
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096').cuda()
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


# Hierarchical attentive reader
class Find(nn.Module):
    def __init__(self, k, emb_dim, att_dim, trans_dim):
        super(Find, self).__init__()
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.trans_dim = trans_dim
        self.softmax = nn.Softmax(dim=-1)
        self.sparsemax = Sparsemax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.topk = k
        self.gru = nn.GRU(emb_dim, trans_dim // 2, batch_first=True, bidirectional=True, dropout=0.1)


        self.Ws1 = nn.Linear(trans_dim, 1, bias=False)
        self.Ws2 = nn.Linear(trans_dim, 1, bias=False)
        self.Ws3 = nn.Linear(trans_dim, 1, bias=False)
        self.Wq1 = nn.Linear(trans_dim, 1, bias=False)
        self.Wq2 = nn.Linear(trans_dim, 1, bias=False)
        self.Wq3 = nn.Linear(trans_dim, 1, bias=False)
        self.Wc1 = nn.Linear(trans_dim, 1, bias=False)
        self.Wc2 = nn.Linear(trans_dim, 1, bias=False)
        self.Wc3 = nn.Linear(trans_dim, 1, bias=False)

        self.Wm1 = nn.Linear(trans_dim, 1, bias=False)
        self.Wm2 = nn.Linear(trans_dim, 1, bias=False)
        self.Wm3 = nn.Linear(trans_dim, 1, bias=False)


        self.Sq = nn.Linear(trans_dim, 1)
        self.Sc = nn.Linear(trans_dim, 1)
        self.Sqd = nn.Linear(trans_dim, 1)
        self.Scd = nn.Linear(trans_dim, 1)

        self.fq1 = nn.Linear(trans_dim, 1, bias=False)
        self.fq2 = nn.Linear(trans_dim, 1, bias=False)
        self.fq3 = nn.Linear(trans_dim, 1, bias=False)
        self.fc1 = nn.Linear(trans_dim, 1, bias=False)
        self.fc2 = nn.Linear(trans_dim, 1, bias=False)
        self.fc3 = nn.Linear(trans_dim, 1, bias=False)

        self.Wfindm = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * trans_dim, 1, bias=False))
        self.Wfindm2 = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * trans_dim, 1, bias=False))


        self.Wcom = nn.Sequential(nn.Dropout(0.1), nn.Linear(trans_dim * 3, trans_dim))
        
    def forward(self, data, flag):
        paras = data['supports']
        query_ent = data['query_ent']
        query = data['query']

        
        # convert paragraphs to sentences
        sents = []
        lens_sents = []
        ll = 50
        nn = 50
        sents = []
        for p in paras:
            tokens = p.split()
            tokens = [' '.join(tokens[l: l+ll]) for l in range(0, len(tokens), ll)]
            sents.extend(tokens)

        # concatenate query with contexts
        sent_tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pi)) for pi in sents]  
        qtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS] ' + query_ent + ' [SEP] '))

        sent_q_tokens = [qtokens + pi for pi in sent_tokens]
        batch_max_len = max([len(s) for s in sent_q_tokens])
        sent_q_tokens = pad_sequences(sent_q_tokens, maxlen=min(batch_max_len, 512), dtype="long", truncating="post", padding="post")
        sent_q_tokens = torch.LongTensor(sent_q_tokens).to(device)
        smask_q = (sent_q_tokens != 0)
        # extract query representations
        qlen = len(qtokens)
        with torch.no_grad():
            qvec = bert_model(sent_q_tokens, attention_mask = smask_q)[0][:, 1: 1 + qlen, :]
        
        qvec = self.gru(qvec)[0]
        self_q = self.softmax(self.Sq(qvec).transpose(1, 2)) # (num_doc, 1, query_len)
        qvec = torch.bmm(self_q, qvec).squeeze(1) # (num_doc, dim)
        self_qvec = torch.matmul(self.softmax(self.Sqd(qvec).transpose(0, 1)), qvec) # (1, dim)

        # concatenate each candidate with contexts
        clens = torch.tensor([len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c))) for c in data['candidates']]) 
        cmask = self.gen_mask(max(clens), clens)
        masked_candidates = torch.zeros(len(sent_tokens), cmask.size(0), cmask.size(1), self.emb_dim).to(device)
        for i, cand, cl in zip(range(len(data["candidates"])), data["candidates"], clens):
            ctokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS] ' + cand + ' [SEP] '))

            sent_c_tokens = [ctokens + pi for pi in sent_tokens]
            batch_max_len = max([len(s) for s in sent_c_tokens])
            sent_c_tokens = pad_sequences(sent_c_tokens, maxlen=min(batch_max_len, 512), dtype="long", truncating="post", padding="post")
            sent_c_tokens = torch.LongTensor(sent_c_tokens).to(device)
            smask_c = (sent_c_tokens != 0)
            with torch.no_grad():
                cvec = bert_model(sent_c_tokens, attention_mask = smask_c)[0][:, 1: 1 + cl, :]

            masked_candidates[:, i, :cl, :] = cvec

        num_doc, num_cand, cand_len, dim = masked_candidates.size()
        masked_candidates = masked_candidates.view(-1, cand_len, dim).contiguous() # (num_doc*num_cand, cand_len, dim)
        masked_candidates = self.gru(masked_candidates)[0].contiguous()

        self_c = self.Sc(masked_candidates).squeeze(-1) # (num_doc*num_cand, cand_len)
        self_c = self.softmax(self_c.masked_fill(cmask.repeat(num_doc, 1), -1e9))
        masked_candidates = torch.bmm(self_c.unsqueeze(1), masked_candidates).squeeze(1).view(num_doc, num_cand, -1).contiguous() # (num_doc, num_cand, dim)
        # (num_cand, dim)
        self_cvecs = torch.bmm(self.softmax(self.Scd(masked_candidates).squeeze(-1).transpose(0, 1)).unsqueeze(1), \
                                masked_candidates.transpose(0, 1)).squeeze(1)


        # compute context representation
        batch_max_len = max([len(s) for s in sent_tokens])
        sent_tokens = pad_sequences(sent_tokens, maxlen=min(batch_max_len, 512), dtype="long", truncating="post", padding="post")
        sent_tokens = torch.LongTensor(sent_tokens).to(device)
        smask = (sent_tokens != 0)

        with torch.no_grad():
            masked_sents = bert_model(sent_tokens, attention_mask = smask)[0]
        masked_sents = self.gru(masked_sents)[0].contiguous()

        Hq = torch.Tensor().to(device)
        Hc = torch.Tensor().to(device)
        Hm = torch.Tensor().to(device)
        Hm2 = torch.Tensor().to(device)

        
        # Hq: (num_doc, dim), Hc: (num_cand, num_doc, dim)
        for ids in range(0, masked_sents.size(0), nn):
            Hq_i, Hc_i, Hscore_q_i = self.context_attention(self_qvec, self_q, self_cvecs, \
                                self_c, masked_sents[ids: ids+nn], cmask, smask[ids: ids+nn])             

            # self attention for relocating intermediate entities
            Hm_i, Hscore_q1_i = self.self_attention(Hq_i, masked_sents[ids: ids+nn], Hscore_q_i, smask[ids: ids+nn])
            # Hm2_i, _ = self.self2_attention(Hm_i, masked_sents[ids: ids+nn], Hscore_q1_i, smask[ids: ids+nn])

            Hq = torch.cat([Hq, Hq_i], dim=0)
            Hc = torch.cat([Hc, Hc_i], dim=1)
            Hm = torch.cat([Hm, Hm_i], dim=0)
            # Hm2 = torch.cat([Hm2, Hm2_i], dim=0)
        
        # hop 0 attention
        q2p_att, c2p_att = self.find_attention(self_qvec, self_q, self_cvecs, self_c, Hq, Hc, cmask)
        q2p = self.Wcom(torch.cat([self_qvec, q2p_att, self_qvec * q2p_att], dim=1))
        c2p = self.Wcom(torch.cat([self_cvecs, c2p_att, self_cvecs * c2p_att], dim=1))

        # 1-hop attention
        v1, score1 = self.findm_attention(q2p, Hm, flag)
        
        # hop 2 attention
        # v2, score2 = self.findm2_attention(q2p, v1, Hm2)

        return q2p, c2p, v1, score1

        


    def find_attention(self, query, self_q, masked_candidates, self_c, key_q, key_c, cmask):
        # query: (num_doc, dim)
        # masked_candidates: (num_doc, num_cand, dim)
        # key_q: (num_doc, dim), key_c: (num_cand, num_doc, dim), cmask: (num_cand, cand_len)

        # (num_key, 1)
        score_q = self.fq1(query) + self.fq2(key_q) + \
                        self.fq3(query * key_q)
        score_q = self.softmax(score_q.transpose(0, 1))
        
        attended_q = torch.matmul(score_q, key_q)

        # (num_cand, num_key)
        score_c = self.fc1(masked_candidates).unsqueeze(1) + self.fc2(key_c) + \
                        self.fc3(masked_candidates.unsqueeze(1) * key_c)
        score_c = score_c.squeeze(-1)
        score_c = self.softmax(score_c)        
        
        attended_c = torch.bmm(score_c.unsqueeze(1), key_c).squeeze(1) # (num_cand, dim)

        return attended_q, attended_c



    def gen_mask(self, max_len, lengths):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num, -1) + 1 
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask



    def context_attention(self, query, self_q, candidates, self_c, key, cmask, kmask):
        # query: (num_doc, dim)
        # candidates: (num_doc, num_cand, dim)
        # key: (num_doc, doc_len, dim), cmask: (num_cand, cand_len)
        # kmask: (num_doc, doc_len)

        # (num_key, key_len, 1)
        score_q = self.Wq1(query).unsqueeze(1) + self.Wq2(key) + \
                        self.Wq3(query.unsqueeze(1) * key)
        score_q = score_q.squeeze(-1)
        score_q = score_q.masked_fill(kmask, -1e9)
        # (num_key, key_len)
        score_q = self.softmax(score_q)

        attended_q = torch.bmm(score_q.unsqueeze(1), key).squeeze(1) # (num_key, dim)
                        
        # (num_cand, num_key, key_len)
        score_c = self.Wc1(candidates).unsqueeze(1).unsqueeze(1) + self.Wc2(key).unsqueeze(0).repeat(candidates.size(0), 1, 1, 1) + \
                        self.Wc3(candidates.unsqueeze(1).unsqueeze(1) * key.unsqueeze(0).repeat(candidates.size(0), 1, 1, 1))
        score_c = score_c.squeeze(-1)
        # (num_cand, num_key, key_len)
        score_c = self.softmax(score_c.masked_fill(kmask.unsqueeze(0).repeat(candidates.size(0), 1, 1), -1e9))
        
        attended_c = torch.einsum('ijk,jkl->ijl', (score_c, key)) # (num_cand, num_key, dim)

        return attended_q, attended_c, score_q


    def self_attention(self, query, key, att_score, kmask):
        # query: (num_key, dim), key: (num_key, key_len, dim), att: (num_key, key_len), kmask: (num_key, key_len)
        num_key, key_len, dim = key.size()
        # (num_key, key_len x key_len, 1)
        score = self.Ws1(query.unsqueeze(1) + key).repeat_interleave(key.size(1), dim=1) + \
                        self.Ws2(key).repeat(1, key.size(1), 1) + \
                        self.Ws3((query.unsqueeze(1) + key).repeat_interleave(key.size(1), dim=1) * key.repeat(1, key.size(1), 1))
        # (num_key, key_len, key_len)
        kkmask = torch.bmm(kmask.unsqueeze(-1).eq(0).float(), kmask.unsqueeze(1).eq(0).float()).eq(0)
        score = score.view(num_key, key_len, key_len).masked_fill(kkmask, -1e9)

        score = self.softmax(score) * att_score.unsqueeze(-1)
        score = torch.sum(score, dim=1) # (num_key, key_len)
        # (num_key, dim)
        attended = torch.bmm(score.unsqueeze(1), key).squeeze(1)

        return attended, score


    def self2_attention(self, query, key, att_score, kmask):
        # query: (num_key, dim), key: (num_key, key_len, dim), att: (num_key, key_len), kmask: (num_key, key_len)
        num_key, key_len, dim = key.size()
        # (num_key, key_len x key_len, 1)
        score = self.Wm1(query.unsqueeze(1) + key).repeat_interleave(key.size(1), dim=1) + \
                        self.Wm2(key).repeat(1, key.size(1), 1) + \
                        self.Wm3((query.unsqueeze(1) + key).repeat_interleave(key.size(1), dim=1) * key.repeat(1, key.size(1), 1))
        # (num_key, key_len, key_len)
        kkmask = torch.bmm(kmask.unsqueeze(-1).eq(0).float(), kmask.unsqueeze(1).eq(0).float()).eq(0)
        score = score.view(num_key, key_len, key_len).masked_fill(kkmask, -1e9)

        score = self.softmax(score) * att_score.unsqueeze(-1)
        score = torch.sum(score, dim=1) # (num_key, key_len)
        # (num_key, dim)
        attended = torch.bmm(score.unsqueeze(1), key).squeeze(1)

        return attended, score



    def findm_attention(self, query, key, flag):

        score1 = self.Wfindm(torch.cat([query.repeat(key.size(0), 1), key, query * key], dim=1))
        score1 = F.sigmoid(score1)

        ind1_topk = torch.topk(score1.view(-1), k=min(self.topk, key.size(0)))[1]
        v1 = key[ind1_topk] # (k, dim)

        return v1, score1[ind1_topk]


    def findm2_attention(self, query, v1, key):
        # v1: (k, dim), key: (num_doc, dim)
        # (k, num_doc)
        score2 = self.Wfindm2(torch.cat([(query+v1).unsqueeze(1).repeat(1, key.size(0), 1), \
                            key.unsqueeze(0).repeat(v1.size(0), 1, 1), \
                            (query+v1).unsqueeze(1).repeat(1, key.size(0), 1) * key.unsqueeze(0).repeat(v1.size(0), 1, 1)], dim=-1))
        

        score2 = F.sigmoid(score2.squeeze(-1))
        max_score, max_ind = torch.max(score2, dim=-1)
        v2 = key[max_ind]

        return v2, max_score.view(-1, 1)



# Multi-hop logic reasoner
class Model(nn.Module):
    def __init__(self, qrel_dic, emb_dim, trans_dim, att_dim, rel_dim, nrel, nclause, nclause1, nclause2, nclause3, ktop):
        super(Model, self).__init__()
        self.find = Find(ktop, emb_dim, att_dim, trans_dim)
        self.qrel_dic = qrel_dic
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.rel_dim = rel_dim
        self.trans_dim = trans_dim
        self.nrel = nrel
        self.nclause = nclause
        self.nclause1 = nclause1
        self.nclause2 = nclause2
        self.nclause3 = nclause3
        self.rel_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nrel, trans_dim, trans_dim)), requires_grad=True)

        self.ktop = ktop

        self.query_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(len(qrel_dic), att_dim)), requires_grad=True)
        self.key_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nrel, att_dim)), requires_grad=True)
        self.pred0_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nclause1, att_dim)), requires_grad=True)
        self.pred1_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nclause2, att_dim)), requires_grad=True)
        self.pred2_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nclause3, att_dim)), requires_grad=True)
        
        self.Wkey = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, nclause * 50)), requires_grad=True)
        self.Wquery = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, nclause * 50)), requires_grad=True)
        self.Wkey_0 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_0 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)

        self.Wkey_l1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wkey_r1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_l1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_r1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wvalue_1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)

        self.Wkey_l2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wkey_m2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wkey_r2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_l2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_m2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wpred_r2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)
        self.Wvalue_2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(att_dim, att_dim)), requires_grad=True)

        self.Wrel = nn.Linear(4 * trans_dim, nrel)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.sparsemax = Sparsemax(dim=-1)

    # use inductive logic to compute the outputs
    def forward(self, data, flag):
        hop0_q, hop0_c, m1, m1score = self.find(data, flag)
        score_0, reg_0 = self.ilp_hop_0(hop0_q, hop0_c)
        score_1, reg_1 = self.ilp_hop_1(hop0_q, hop0_c, m1, m1score)
        # score_2, reg_2 = self.ilp_hop_2(hop0_q, hop0_c, m1, m2, m2score)


        final_score, entropy = self.ilp_all(score_0, score_1, self.qrel_dic[data['name']])
        return final_score
    
    def ilp_all(self, score_0, score_1, q):
        # (nclause1+nclause2+nclause3) * att_dim
        pred_emb = torch.cat((self.pred0_emb, self.pred1_emb), dim=0)
        # 1 * att_dim
        query_emb = self.query_emb[q].unsqueeze(0)
        # nclause * (nclause1+nclause2+nclause3)
        Wclause = torch.bmm(torch.mm(query_emb, self.Wquery).view(self.nclause, 1, -1), \
                            torch.mm(pred_emb, self.Wkey).view(self.nclause, -1, pred_emb.size(0))).squeeze(1)

        Wclause = self.sparsemax(Wclause)
        entropy = Categorical(probs = Wclause).entropy()
        score = torch.cat((score_0, score_1), dim=1)
        final_scores = torch.exp(torch.matmul(Wclause, torch.log((score + 1e-6).transpose(0, 1))))

        final_scores = torch.max(final_scores, dim=0)[0]

        return final_scores, entropy.sum()


    def ilp_hop_0(self, qemb, cemb):
        # size (npair, nrel)
        rel_scores = self.score(qemb.repeat(cemb.size(0), 1), cemb)
        

        Wclause = torch.matmul(torch.mm(self.pred0_emb, self.Wpred_0), torch.mm(self.key_emb, self.Wkey_0).transpose(0,1)) # size (npred, nkey)
        Wclause = self.sparsemax(Wclause)

        sym = torch.mm(Wclause, torch.t(Wclause))
        sym -= torch.eye(Wclause.shape[0]).to(device)
        orth_loss = sym.abs().sum()

        # size (bq, npred)
        clause_scores = torch.exp(torch.matmul(Wclause, torch.log((rel_scores + 1e-6).transpose(0, 1))))
        clause_scores = clause_scores.transpose(0, 1) # size (npair, nclause)
        
        return clause_scores, orth_loss

    
    def ilp_hop_1(self, qemb, cemb, m1emb, m1score):
        # find intermediate entities
        scores_all = []

        Wclause_l = torch.matmul(torch.mm(self.pred1_emb, self.Wpred_l1), torch.mm(self.key_emb, self.Wkey_l1).transpose(0,1)) # size (npred, nkey)
        Wclause_l = self.sparsemax(Wclause_l)
        new_query_emb = self.pred1_emb + torch.matmul(Wclause_l, torch.mm(self.key_emb, self.Wvalue_1))

        sym = torch.mm(Wclause_l, torch.t(Wclause_l))
        sym -= torch.eye(Wclause_l.shape[0]).to(device)
        orth_loss_l = sym.abs().sum()
        

        Wclause_r = torch.matmul(torch.mm(new_query_emb, self.Wpred_r1), torch.mm(self.key_emb, self.Wkey_r1).transpose(0,1)) # size (npred, nkey)
        Wclause_r = self.sparsemax(Wclause_r)

        sym = torch.mm(Wclause_r, torch.t(Wclause_r))
        sym -= torch.eye(Wclause_r.shape[0]).to(device)
        orth_loss_r = sym.abs().sum()


        # compute clause scores for hop-1 reasoning
        # (topk, nc, nkey)
        rel1_scores = self.score(qemb.repeat(m1emb.size(0), 1), m1emb)
        rel1_scores = rel1_scores.repeat_interleave(cemb.size(0), dim=0)
        # (topk*nc, nkey)
        rel2_scores = self.score(m1emb.unsqueeze(1).repeat(1, cemb.size(0), 1), \
                            cemb.unsqueeze(0).repeat(m1emb.size(0), 1, 1)).view(-1, rel1_scores.size(-1)).contiguous()

    
        # size (nclause, topk*nc)
        clause_scores = (torch.exp(torch.matmul(Wclause_l, torch.log((rel1_scores + 1e-6).transpose(0, 1)))) * \
                            torch.exp(torch.matmul(Wclause_r, torch.log((rel2_scores + 1e-6).transpose(0, 1))))) ** 0.5
        clause_scores = torch.min(torch.ones(clause_scores.size()).to(device), clause_scores)
        clause_scores = clause_scores.view(clause_scores.size(0), -1, cemb.size(0)).contiguous()

        # size (nclause, nc)
        clause_scores = torch.max(clause_scores * m1score.transpose(0, 1).unsqueeze(-1), dim=1)[0]

        return clause_scores.view(-1, self.nclause2), orth_loss_l + orth_loss_r
    
    
    def ilp_hop_2(self, qemb, cemb, m1, m2, score):

        # find intermediate entities
        Wclause_l = torch.matmul(torch.mm(self.pred2_emb, self.Wpred_l2), torch.mm(self.key_emb, self.Wkey_l2).transpose(0,1)) # size (npred, nkey)
        Wclause_l = self.sparsemax(Wclause_l)
        query_emb_l = self.pred2_emb + torch.matmul(Wclause_l, torch.mm(self.key_emb, self.Wvalue_2))

        sym = torch.mm(Wclause_l, torch.t(Wclause_l))
        sym -= torch.eye(Wclause_l.shape[0]).to(device)
        orth_loss_l = sym.abs().sum()

        Wclause_m = torch.matmul(torch.mm(query_emb_l, self.Wpred_m2), torch.mm(self.key_emb, self.Wkey_m2).transpose(0,1)) # size (npred, nkey)
        Wclause_m = self.sparsemax(Wclause_m)
        query_emb_m = query_emb_l + torch.matmul(Wclause_m, torch.mm(self.key_emb, self.Wvalue_2))


        sym = torch.mm(Wclause_m, torch.t(Wclause_m))
        sym -= torch.eye(Wclause_m.shape[0]).to(device)
        orth_loss_m = sym.abs().sum()

        Wclause_r = torch.matmul(torch.mm(query_emb_m, self.Wpred_r2), torch.mm(self.key_emb, self.Wkey_r2).transpose(0,1)) # size (npred, nkey)
        Wclause_r = self.sparsemax(Wclause_r)

        sym = torch.mm(Wclause_r, torch.t(Wclause_r))
        sym -= torch.eye(Wclause_r.shape[0]).to(device)
        orth_loss_r = sym.abs().sum()

        # (topk, nkey)
        rel1_scores = self.score(qemb.repeat(m1.size(0), 1), m1)
        rel1_scores = rel1_scores.repeat_interleave(cemb.size(0), dim=0)
        rel2_scores = self.score(m1, m2)
        rel2_scores = rel2_scores.repeat_interleave(cemb.size(0), dim=0)
        # (topk*nc, nkey)
        rel3_scores = self.score(m2.unsqueeze(1).repeat(1, cemb.size(0), 1), \
                            cemb.unsqueeze(0).repeat(m2.size(0), 1, 1)).view(-1, rel1_scores.size(-1)).contiguous()

        # size (nclause, topk*nc)
        clause_scores = (torch.exp(torch.matmul(Wclause_l, torch.log((rel1_scores + 1e-6).transpose(0, 1)))) * \
                            torch.exp(torch.matmul(Wclause_m, torch.log((rel2_scores + 1e-6).transpose(0, 1)))) * \
                            torch.exp(torch.matmul(Wclause_r, torch.log((rel3_scores + 1e-6).transpose(0, 1))))) ** (1.0/3)
        clause_scores = torch.min(torch.ones(clause_scores.size()).to(device), clause_scores)
        clause_scores = clause_scores.view(clause_scores.size(0), -1, cemb.size(0))
        # (nclause, nc)
        clause_scores = torch.max(clause_scores * score.transpose(0, 1).unsqueeze(-1), dim=1)[0]

        return clause_scores.view(-1, self.nclause3), orth_loss_l + orth_loss_m + orth_loss_r



    def score(self, e1_emb, e2_emb):

        rel_score = self.softmax(self.Wrel(self.dropout(F.tanh(torch.cat([e1_emb, e2_emb, e1_emb - e2_emb, e1_emb * e2_emb], dim=-1)))))
        return rel_score



def train(data_train, optimizer, scheduler, config, flag=True):
    loss_epoch = 0.0
    for batch_ind, batch_data in enumerate(data_train):   
        loss_batch = 0.0
        for data in batch_data:       
            scores = model(data, flag)
            loss_batch += compute_loss(scores, data['label'], config["entropy_coeff"])
        loss_batch = loss_batch / len(batch_data)

        if not torch.isnan(loss_batch).any():
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss_epoch += loss_batch.item()

            print('Batch {}: main loss:{:.2f}'.format(batch_ind, loss_batch.item()))

    scheduler.step()
    return loss_epoch


def compute_loss(predictions, l, c_entropy):
    predictions = predictions.view(-1, 1)
    predictions = torch.cat((1.0 - predictions, predictions), dim=1)
    labels = torch.zeros((predictions.size(0)), dtype=torch.long)
    labels[l] = 1
    loss_flat = -torch.log(torch.gather(predictions.contiguous(), dim=1, \
                                        index=torch.LongTensor(labels).contiguous().view(-1, 1).to(device)))
    loss_flat[torch.isnan(loss_flat)] = 0
    loss = loss_flat.sum() / loss_flat.nonzero().size(0)# + c_entropy * entropy

    return loss

def compute_loss_ce(predictions, l, c_entropy):
    predictions = predictions.view(-1)
    loss = -torch.log(predictions[l])

    return loss

def compute_loss_mll(predictions, label, c_entropy):
    predictions = predictions.view(-1)
    if label < predictions.size(0) - 1:
        neg_predictions = torch.cat([predictions[:label], predictions[label+1:]], dim=0)
    else:
        neg_predictions = predictions[:label]
    pos_prediction = predictions[label]
    loss = -torch.log(pos_prediction) - torch.log(1.0 - torch.max(neg_predictions))

    return loss


def compute_loss_mm(predictions, label, c_entropy):
    predictions = predictions.view(-1)
    # mm = nn.MultiLabelMarginLoss()
    # label = torch.nonzero(torch.LongTensor(labels)).item()
    # loss = mm(predictions, y) + c_entropy * entropy

    neg_predictions = torch.cat([predictions[:label], predictions[label+1:]], dim=0)
    pos_prediction = predictions[label]
    loss = torch.mean(torch.max(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device) + neg_predictions - pos_prediction))

    return loss# + c_entropy * entropy


def predict(data_test, flag=False):
    correct = 0
    count = 0
    for ind, data in enumerate(data_test):
        count += 1
        scores = model(data, flag)
        if torch.argmax(scores).item() == data['label']:
            correct += 1
    
    acc = float(correct) / count
    print("Accuracy: ", acc)
    f_out.write("Accuracy: " + str(acc) + "\n")
    f_out.write("correct-count: " + str(correct) + "-" + str(count))
    return acc


def clean_token(word):
    word = word.replace("''", '"')
    word = word.replace("``", '"')
    # word = word.replace("-", " ")
    word = word.replace(" - ", "-")
    word = word.replace("_", " ")
    return word


def process_file(datalist, name):
    new_data = []
    for d in datalist: 
        d['supports'] = [clean_token(d).lower() for d in d['supports']]
        if len(d['query'].split()) > 1:
            d["answer"] = clean_token(d['answer']).lower()
            d["candidates"] = [clean_token(cand.lower()) for cand in d['candidates']]
            d["query_ent"] = clean_token(' '.join(d['query'].split()[1:]).lower())
            d["query"] = clean_token(d["query"])
            d["name"] = name.replace('_', ' ').lower()
            d["label"] = d["candidates"].index(d["answer"])
            new_data.append(d)

    return new_data


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


if __name__ == "__main__":

    folder_dev = os.fsencode("data/wikihop/dev_split")
    folder_train = os.fsencode("data/wikihop/train_split")

    train_all, test_all = [], []
    qrel_all = OrderedDict()

    filename='country'
    if os.fsencode('train_' + filename + '.json') in os.listdir(folder_train):
        print("Working on query: ", filename)
        qrel_all[filename.replace('_', ' ')] = len(qrel_all)
        with open("data/wikihop/dev_split/dev_" + filename + '.json', "r") as f:
            test_data = json.load(f)
            test_data = process_file(test_data, filename)
            test_all.extend(test_data) 
        with open("data/wikihop/train_split/train_" + filename + '.json', 'r') as f:
            train_data = json.load(f)
            train_data = process_file(train_data, filename)
            train_all.extend(train_data)
        print("Data Loaded..")
    
    random.seed(111)
    random.shuffle(train_all)
        
    config = {
                "dim": 768,
                "transformed_dim": 256,
                "att_dim": 100,
                "num_rel": 10,
                "rel_dim": 100,
                "epoch": 15,
                "lr": 1e-3,
                "batch_size": 10,
                "nclause": 5,
                "nclause1": 5,
                "nclause2": 5,
                "nclause3": 5,
                "topk": 5,
                "entropy_coeff": 0.01
    }
    

    model = Model(qrel_all, config["dim"], config["transformed_dim"], config["att_dim"], config["rel_dim"], config["num_rel"], \
                    config["nclause"], config["nclause1"], config["nclause2"], config["nclause3"], config["topk"])
    
    model = model.to(device)   
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = CosineWithRestarts(optimizer, T_max=5, factor=2, eta_min=1e-5)


    for i in range(config["epoch"]):

        model.train()

        data_train = [train_all[i : i + config["batch_size"]] for i in range(0, len(train_all), config["batch_size"])]
        loss_epoch = train(data_train, optimizer, scheduler, config)
        print('Epoch ' + str(i) + ' loss: ', loss_epoch)

        model.eval()
        predict(test_all)

    
                     
                            
                        

        
        