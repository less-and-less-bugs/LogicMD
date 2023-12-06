import torch
import torch.nn as nn
from models.component import TextEncoder, ImageEncoder, GraphConvolution, MCO
from torch.nn.init import xavier_uniform_
from sparsemax import Sparsemax

# from utils import L2_norm, cosine_distance
# from transformers import BertModel
# import torch_geometric.nn.conv as conv

# from utils.data_utils import pad_tensor
"""
assume we have obtained the embedding of tokens, deps, visual objects, object positions i
question: how to incorporate obvious rules
"""
class RFND(nn.Module):
    def __init__(self, input_size=768, out_size=300, rnn=False, rnn_type='LSTM', ch=False, finetune=True,
                 instance_dim=200, top_K=10, size_clues=[5, 5, 5, 5, 5], relation_dim=200, ans_number=2, answer_dim=200,
                 guide_head=5, norm_type='T', threshold=0.4, rate=0.2, gcnumber=1):
        super(RFND, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.rnn = rnn
        self.rnn_type = rnn_type
        self.ch = ch
        self.finetune = finetune
        self.instance_dim = instance_dim
        self.top_K = top_K
        self.size_clues = size_clues
        self.relation_dim = relation_dim
        self.ans_number = ans_number
        self.answer_dim = answer_dim
        # self.H_clause = H_clause
        # easy to set parameter
        self.out_size = self.instance_dim
        self.relation_dim = self.instance_dim
        self.answer_dim = self.instance_dim

        self.guide_head = guide_head
        self.norm_type = norm_type
        self.threshold = threshold
        self.rate = rate
        self.gcnumber = gcnumber

        self.answer_set = nn.Parameter(torch.Tensor(self.ans_number, self.answer_dim))
        # our framework
        self.text_encoder = TextEncoder(input_size=self.input_size, out_size=self.out_size, rnn=self.rnn,
                                        rnn_type=self.rnn_type, ch=self.ch, finetune=self.finetune)
        self.img_encoder = ImageEncoder(input_dim=768, inter_dim=500, output_dim=self.out_size)
        # may be replaced by more powerful GCN
        self.gc = nn.ModuleList([GraphConvolution(self.out_size, self.out_size), GraphConvolution(self.out_size, self.out_size)])
        # self.gc1 = GraphConvolution(self.out_size, self.out_size)
        # self.gc2 = GraphConvolution(self.out_size, self.out_size)

        # self.norm = nn.LayerNorm(self.out_size)

        self.patch_score = nn.Linear(self.out_size, 1)
        self.token_score = nn.Linear(self.out_size, 1)

        # for left instance generation
        self.linear_T = nn.Sequential(nn.Linear(self.out_size, self.instance_dim), nn.ReLU())
        self.linear_V = nn.Sequential(nn.Linear(self.out_size, self.instance_dim), nn.ReLU())
        self.linear_T_T = nn.Sequential(nn.Linear(self.out_size * 4, 2*self.instance_dim), nn.ReLU(), nn.Linear(2*self.instance_dim, self.instance_dim), nn.ReLU())
        self.linear_T_V = nn.Sequential(nn.Linear(self.out_size * 4, 2*self.instance_dim), nn.ReLU(), nn.Linear(2*self.instance_dim, self.instance_dim), nn.ReLU())
        self.linear_V_V = nn.Sequential(nn.Linear(self.out_size * 4, 2*self.instance_dim), nn.ReLU(), nn.Linear(2*self.instance_dim, self.instance_dim), nn.ReLU())
        # self.linear_W = nn.Linear(self.input_size, self.out_size)
        self.linear_T_top_k = nn.Sequential(nn.Linear(self.instance_dim, 1), nn.ReLU())
        self.linear_V_top_k = nn.Sequential(nn.Linear(self.instance_dim, 1), nn.ReLU())
        self.linear_T_T_top_k = nn.Sequential(nn.Linear(self.instance_dim, 1), nn.ReLU())
        self.linear_T_V_top_k = nn.Sequential(nn.Linear(self.instance_dim, 1), nn.ReLU())
        self.linear_V_V_top_k = nn.Sequential(nn.Linear(self.instance_dim, 1), nn.ReLU())

        # clues set
        self.clues_T = nn.Parameter(torch.Tensor(self.size_clues[0], self.relation_dim))
        self.clues_V = nn.Parameter(torch.Tensor(self.size_clues[1], self.relation_dim))
        self.clues_T_T = nn.Parameter(torch.Tensor(self.size_clues[3], self.relation_dim))
        self.clues_T_V = nn.Parameter(torch.Tensor(self.size_clues[2], self.relation_dim))
        self.clues_V_V = nn.Parameter(torch.Tensor(self.size_clues[4], self.relation_dim))
        # for binding clues to find relation for each answer and instance pair
        self.linear_clue_1 = nn.Linear(in_features=self.relation_dim, out_features=self.instance_dim)
        self.linear_clue_2 = nn.Linear(in_features=self.relation_dim, out_features=self.instance_dim)
        self.linear_instance_1 = nn.Linear(in_features=self.instance_dim, out_features=self.instance_dim)
        self.linear_answer_1 = nn.Linear(in_features=self.answer_dim, out_features=self.instance_dim)
        self.linear_instance_2 = nn.Linear(in_features=self.instance_dim, out_features=self.instance_dim)
        self.linear_answer_2 = nn.Linear(in_features=self.answer_dim, out_features=self.instance_dim)
        # for clause generation
        self.inter_predicates = MCO(input_size=self.instance_dim, nhead=5, dim_feedforward=2*self.instance_dim, dropout=0.2)
        self.guide = nn.ModuleList([nn.Linear(in_features=instance_dim, out_features=1) for i in range(self.guide_head*2)])

        self.atoms_generation = nn.Sequential(
            nn.Linear(in_features=self.instance_dim * 4, out_features=self.instance_dim * 2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=self.instance_dim * 2, out_features=1), nn.Sigmoid())

        self.instance_generation = nn.Sequential(nn.Linear(in_features=self.instance_dim * 4,
                                                           out_features=2*self.instance_dim),
                                                 nn.Dropout(p=0.2),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=self.instance_dim * 2,
                                                           out_features=self.instance_dim), nn.ReLU())
        # add mlp
        self.ans_score_generation = nn.Sequential(nn.Linear(in_features=4*self.instance_dim, out_features=2*self.instance_dim),
                                                  nn.ReLU(), nn.Linear(in_features=2*self.instance_dim, out_features=1))

        self.sparsemax = Sparsemax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()
        # answer set answer_set[0] answer_set[1]
        self._reset_parameters()

    def forward(self, imgs, encoded_texts, mask_batch_T, mask_batch_TT,
                mask_batch_TV, adj_matrix, word_len,  word_spans):
        """generate instance for each predicate"""
        # (N,r,outdim)
        imgs_0 = self.img_encoder(imgs)
        # (N, L, outdim)
        texts_0 = self.text_encoder(encoded_texts, word_spans, word_len)
        r = imgs_0.size(1)
        l = texts_0.size(1)
        # captions_0 = self.text_encoder(encoded_captions, cap_word_spans, cap_word_lens)
        # (N, L+r, outdim)

        gc_clause_list = []
        for gc_i in range(self.gcnumber):
            # (N,L,D)
            E_T_top_k, E_V_top_k, E_T_T_top_k, E_T_V_top_k, E_V_V_top_k = self.generate_topk_instance(texts=texts_0,
                                                                                                      imgs=imgs_0,
                                                                                                      mask_T=mask_batch_T,
                                                                                                      mask_TT=mask_batch_TT,
                                                                                                  mask_TV=mask_batch_TV)
            # N, self.top_k, A，self.instance_dim
            clues_T = self.bind_relation_instance(E_top_k=E_T_top_k, clues=self.clues_T)
            clues_V = self.bind_relation_instance(E_top_k=E_V_top_k, clues=self.clues_V)
            clues_T_T = self.bind_relation_instance(E_top_k=E_T_T_top_k, clues=self.clues_T_T)
            clues_T_V = self.bind_relation_instance(E_top_k=E_T_V_top_k, clues=self.clues_T_V)
            clues_V_V = self.bind_relation_instance(E_top_k=E_V_V_top_k, clues=self.clues_V_V)
            # L = 5*self.top_K
            # (N,L,A, D)
            E = torch.cat([E_T_top_k, E_V_top_k, E_T_T_top_k, E_T_V_top_k, E_V_V_top_k], dim=1).contiguous().cuda().unsqueeze(2)\
                .expand(-1, -1, self.ans_number, -1)
            # L = 5*self.top_K
            # N, L, A，self.instance_dim
            predicates = torch.cat([clues_T, clues_V, clues_T_T, clues_T_V,  clues_V_V], dim=1)
            # N, L, A
            del clues_T, clues_V, clues_T_T, clues_T_V,  clues_V_V,E_T_top_k, E_V_top_k, E_T_T_top_k, E_T_V_top_k, E_V_V_top_k
            torch.cuda.empty_cache()
            instance_score_by_ans = self.find_instance_by_answer(E)
            # N, L, A，self.instance_dim
            predicates_set = []
            for i in range(self.ans_number):
                predicates_ = self.inter_predicates(tgt=predicates[:, :, i, :], src=predicates[:, :, i, :], src_key_padding_mask=None)
                predicates_set.append(predicates_)
            predicates_ = torch.stack(predicates_set, dim=2)
            # A, N, L
            # E = E * self.answer_set
            atoms = self.atoms_generation(torch.cat([E, predicates, E - predicates, E * predicates],  dim=-1)).squeeze().permute(2, 0, 1)
            # N, L
            I_score = self.softmax(self.patch_score(imgs_0).squeeze())
            T_score = self.softmax(self.token_score(texts_0).squeeze())
            # N, D
            I_emb = torch.bmm(imgs_0.permute(0, 2, 1), I_score.unsqueeze(2)).squeeze()
            T_emb = torch.bmm(texts_0.permute(0, 2, 1), T_score.unsqueeze(2)).squeeze()
            # N,instance_dim
            E_Image_Text = self.linear_T_V(torch.cat([T_emb, I_emb, T_emb*I_emb, T_emb-I_emb], dim=-1))
            # self.guide_num, l
            multiple_guide_clauses = []
            for i in range(self.guide_head):
                # (N,L,A) remove instance_score_by_ans*instance_score_by_ans or image+instance+answer
                # N, L, A
                final_choose_score = self.sparsemax((self.guide[2*i](predicates_).squeeze())*(
                    self.guide[2*i+1](E_Image_Text).unsqueeze(2).expand(-1, predicates_.size(1), predicates_.size(2)))) * self.sparsemax(instance_score_by_ans)
                # (A,N,L)
                final_choose_score = self.sparsemax(final_choose_score.permute(2, 0, 1))
                # print(torch.max(final_choose_score, dim=-1))
                # find atoms that the score is higher than self.threshold.
                # (A,N,L)
                flag_ = torch.ge(final_choose_score, self.threshold)
                # may save for analysis
                logic_score = []
                for j in range(flag_.size(0)):
                    flag = flag_[j]
                    for i in range(flag.size(0)):
                        tmp = atoms[j][i][flag[i]]
                        if tmp.size(0) == 0:
                            # null find top k
                            value, index = self.find_top_k_atoms(final_choose_score[j][i])
                            tmp = torch.index_select(atoms[j][i], 0, index)*value
                        # logic reasoning for per sample per head
                        logic_score.append(tmp)
                # ansnumber*N
                logic_score = self.conjunction_logic_reasoning(logic_score)
                # H, N
                multiple_guide_clauses.append(logic_score)
            # 2N，5
            multiple_guide_clauses = torch.stack(multiple_guide_clauses, dim=1)
            # the maximum probability o 2N
            multiple_guide_clauses = torch.max(multiple_guide_clauses, dim=1)[0]
            # 2, N
            gc_clause_list.append(torch.stack(multiple_guide_clauses.chunk(2), dim=0))
            if gc_i != len(self.gc):
                graph = torch.cat([texts_0, imgs_0], dim=1).cuda()
                graph = self.gc[gc_i](graph, adj_matrix)
                texts_0 = graph[:, :l, :]
                imgs_0 = graph[:, l:, :]
        # further to compute the probability
        # list Gc_number_layer, 2, N
        return gc_clause_list

            # logic
    def generate_topk_instance(self, texts, imgs, mask_T, mask_TT, mask_TV):
        # (N, L, O)
        E_T = self.linear_T(texts)
        E_V = self.linear_V(imgs)
        l = texts.size()[1]
        r = imgs.size()[1]
        # N,L,1,D
        texts_1 = texts.unsqueeze(2)
        # N,1,L,D
        texts_2 = texts.unsqueeze(1)
        # N,r,1,D
        imgs_1 = imgs.unsqueeze(2)
        # N,1,r,D
        imgs_2 = imgs.unsqueeze(1)
        texts_T_T_1 = texts_1.expand(-1, -1, l, -1)
        texts_T_T_2 = texts_2.expand(-1, l, -1, -1)
        texts_T_V_1 = texts_1.expand(-1, -1, r, -1)
        imgs_T_V_2 = imgs_2.expand(-1, l, -1, -1)
        imgs_V_V_1 = imgs_1.expand(-1, -1, r, -1)
        imgs_V_V_2 = imgs_2.expand(-1, r, -1, -1)
        # the first token to all tokens N L L D
        E_T_T = self.linear_T_T(
            torch.cat([texts_T_T_1, texts_T_T_2, texts_T_T_1 * texts_T_T_2, texts_T_T_1 - texts_T_T_2], dim=-1))
        E_T_V = self.linear_T_V(
            torch.cat([texts_T_V_1, imgs_T_V_2, texts_T_V_1 * imgs_T_V_2, texts_T_V_1 - imgs_T_V_2], dim=-1))
        E_V_V = self.linear_V_V(
            torch.cat([imgs_V_V_1, imgs_V_V_2, imgs_V_V_1 * imgs_V_V_2, imgs_V_V_1 - imgs_V_V_2], dim=-1))
        E_T_T = E_T_T.view(E_T_T.size(0), -1, E_T_T.size(3))
        E_T_V = E_T_V.view(E_T_V.size(0), -1, E_T_V.size(3))
        E_V_V = E_T_V.view(E_V_V.size(0), -1, E_V_V.size(3))
        # Find Top_K  need add mask to keep semantic dependency N,L
        E_T_score = self.linear_T_top_k(E_T).squeeze().masked_fill_(mask_T, float("-Inf"))
        # may add mask afterwards
        E_V_score = self.linear_V_top_k(E_V).squeeze()
        mask_TT = mask_TT.view(E_T_T.size(0), -1)
        mask_TV = mask_TV.view(E_T_V.size(0), -1)
        E_T_T_score = self.linear_T_T_top_k(E_T_T).squeeze().masked_fill_(mask_TT, float("-Inf"))
        E_T_V_score = self.linear_T_V_top_k(E_T_V).squeeze().masked_fill_(mask_TV, float("-Inf"))
        # may add mask afterwards
        E_V_V_score = self.linear_V_V_top_k(E_V_V).squeeze()
        E_T_top_k, _ = self.find_top_k(score=E_T_score, E=E_T, mask=None)
        E_V_top_k, _ = self.find_top_k(score=E_V_score, E=E_V, mask=None)
        E_T_T_top_k, _ = self.find_top_k(score=E_T_T_score, E=E_T_T, mask=None)
        E_T_V_top_k, _ = self.find_top_k(score=E_T_V_score, E=E_T_V, mask=None)
        E_V_V_top_k, _ = self.find_top_k(score=E_V_V_score, E=E_V_V, mask=None)

        return E_T_top_k, E_V_top_k, E_T_T_top_k, E_T_V_top_k, E_V_V_top_k

    def find_top_k(self, score, E, mask):
        """

        :param score:
        :param E:
        :param mask:
        :return:
        """

        N, L = score.shape
        if L < self.top_K:
            # pad for subsequent
            E_ = torch.cat([E*score, torch.zeros((N, self.top_K - L, E.size(2)))]).cuda()
        else:
            # find top_K_T obtain index
            values, tok_K_T = torch.topk(score, self.top_K)
            E_ = []
            for i in range(len(tok_K_T)):
                E_.append(torch.index_select(E[i], 0, tok_K_T[i]))
            E_ = torch.stack(E_, dim=0).contiguous() * values
        mask = None
        return E_, mask

    def find_top_k_atoms(self, score):
        """

        :param score (L):
        :param E:
        :return:
        """
        L = score.size(0)
        top_k = int(L * self.rate)
        # find top_K_T
        values, tok_K_T = torch.topk(score, top_k, dim=0)
        return values, tok_K_T

    def bind_relation_instance(self, E_top_k, clues):
        """
        generate atoms consisting of predicate, instance
        :param E_top_k: (N, L, self.instance_dim). L is the number of candidate instances.
        :param clues: (self.size_clues, self.relation_dim). self.size_clues is the number of clues for each instance.
        :param self.answer_set: (answer_number, self.instance_dim). self.size_clues is the number of clues for each instance.
        :return:  (N, L, A，self.instance_dim)
        """
        #  (N, L, A，self.instance_dim)
        E_top_k_1 = self.linear_instance_2(E_top_k).unsqueeze(2).expand(-1, -1, self.ans_number, -1)
        #  (N, L, A，self.instance_dim)
        answer_set = self.linear_answer_2(self.answer_set).unsqueeze(0).unsqueeze(0).expand(E_top_k.size(0),
                                                                                            E_top_k.size(1), -1, -1)
        #  (N, L, A，self.instance_dim)
        instance_whole = self.instance_generation(
            torch.cat([E_top_k_1, answer_set, E_top_k_1 * answer_set, E_top_k_1 - answer_set], dim=-1))
        # (instance_dim, number_clues)
        clues_1 = self.linear_clue_1(clues).transpose(0, 1)
        # (N, L, A，number_clues))
        score_clues = self.sparsemax(instance_whole@clues_1)
        # (N, L, A，self.relation_dim)
        predicates = score_clues@clues
        # (N, L, A，self.instance_dim) the final relationship
        predicates = self.linear_clue_2(predicates)
        return predicates

    def find_instance_by_answer(self, E_top_k):
        """

        :param E_top_k: the concatenation of E_top_k for all [32, 50, 2, 200])
        :return instance_score_by_ans: (N, L, A)
        """
        #  (N, L, A，self.instance_dim)
        E_top_k_1 = self.linear_instance_1(E_top_k)
        #  (N, L, A，self.instance_dim)
        answer_set = self.linear_answer_1(self.answer_set).unsqueeze(0).unsqueeze(0).expand(E_top_k.size(0),

                                                                                            E_top_k.size(1), -1, -1)
        #  (N, L, A)


        instance_score_by_ans = self.ans_score_generation(torch.cat([E_top_k_1, answer_set, E_top_k_1-answer_set,
                                                                     E_top_k_1*answer_set], dim=-1)).squeeze()
        return instance_score_by_ans

    def _reset_parameters(self):
        xavier_uniform_(self.clues_T)
        xavier_uniform_(self.clues_V)
        xavier_uniform_(self.clues_T_V)
        xavier_uniform_(self.clues_T_T)
        xavier_uniform_(self.clues_V_V)
        xavier_uniform_(self.answer_set)
        # for i in range(self.guide):
        #     xavier_uniform_(self.guide[i])

    def conjunction_logic_reasoning(self, logic_score):
        """

        :param logic_score:
        :return: (N) dimension tensor
        """
        logic_score_ = []
        if self.norm_type == 'minimum':
            for sample in logic_score:
                logic_score_.append(torch.min(sample).unsqueeze(0))
        elif self.norm_type == 'product':
            for sample in logic_score:
                # sample is a d-dimension vector
                tmp = sample[0]
                for i in range(1, sample.size(0)):
                    tmp = tmp*sample[i]
                logic_score_.append(tmp.unsqueeze(0))
        elif self.norm_type == 'lukasiewicz':
            for sample in logic_score:
                # sample is a d-dimension vector
                tmp = sample[0]
                for i in range(1, sample.size(0)):
                    tmp = torch.max(torch.tensor([0, tmp + sample[i] -1])).unsqueeze(0)
                logic_score_.append(torch.tensor(tmp).cuda())
        else:
            print("error norm type")
            exit(0)
        logic_score_ = torch.cat(logic_score_, dim=0).cuda().squeeze()
        return logic_score_


def disconjunction_logic_reasoning(logic_score, norm_type):
    logic_score_ = []
    if norm_type == 'minimum':
        for sample in logic_score:
            logic_score_.append(torch.max(sample))
    elif norm_type == 'product':
        for sample in logic_score:
            # sample is a d-dimension vector
            tmp = sample[0]
            for i in range(1, sample.size(0)):
                tmp = 1 - (1 - tmp) * (1-sample[i])
            logic_score_.append(tmp.unsqueeze(0))
    elif norm_type == 'lukasiewicz':
        for sample in logic_score:
            # sample is a d-dimension vector
            tmp = sample[0]
            for i in range(1, sample.size(0)):
                tmp = torch.min(torch.tensor([1, tmp + sample[i]]))
            logic_score_.append(tmp)
    else:
        print("error norm type")
        exit(0)
    logic_score_ = torch.cat(logic_score_, dim=0).cuda().squeeze()
    return logic_score_


def calculate_probability(gc_clause_list, hop, normtype):
    """
    :param gc_clause_list: list of (2, N)
    :param hop: choosed gc hop
    :param normtype: 'minimum', 'product', 'lukasiewicz'
    :return: (N, 2) float tensor cuda
    """
    if hop == 'one':
        return gc_clause_list[0].permute(1, 0)
    elif hop == 'two':
        return gc_clause_list[1].permute(1, 0)
    elif hop == 'three':
        return gc_clause_list[2].permute(1, 0)
    elif hop == 'onetotwo':
        # 2, N, 2
        score = torch.stack([gc_clause_list[0], gc_clause_list[1]], dim=2)
        final_score = []
        for i in range(score.size(0)):
            # N
            final_score.append(disconjunction_logic_reasoning(score[i], normtype))
        # N,2
        return torch.stack(final_score, dim=1)
    elif hop =='onetothree':
        score = torch.stack([gc_clause_list[0], gc_clause_list[1], gc_clause_list[2]], dim=2)
        final_score = []
        for i in range(score.size(0)):
            # N, 2
            final_score.append(disconjunction_logic_reasoning(score[i], normtype))
        return torch.stack(final_score, dim=1)
    elif hop =='twotothree':
        score = torch.stack([gc_clause_list[1], gc_clause_list[2]], dim=2).cuda()
        final_score = []
        for i in range(score.size(0)):
            # N, 2
            final_score.append(disconjunction_logic_reasoning(score[i], normtype))
        return torch.stack(final_score, dim=1)

