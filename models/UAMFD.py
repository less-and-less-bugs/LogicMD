import torch
import torch.nn as nn
from models.component import TextEncoder, ImageEncoder, GraphConvolution, MCO


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (torch.sum((x.permute([1, 0]) - self.mu(x)).permute([1, 0]) ** 2, (1)) + 0.000000023) / (x.shape[1]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean / x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1) * (x_norm + mu.squeeze(1))).permute([1, 0])


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class TokenAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            # SimpleGate(dim=2),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        """
        :param inputs: (N, L, Dim)
        :return: (N, Dim)
        """
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class UAMFD_Net(nn.Module):
    def __init__(self, thresh=0.5, ch=False):
        # NOTE: NOW WE ONLY SUPPORT BASE MODEL!
        self.thresh = thresh
        model_size = 'base'
        self.ch = ch
        print("we are using adaIN")

        self.unified_dim, self.text_dim = 200, 768
        out_dim = 2
        self.num_expert = 2  # 2
        self.depth = 1  # 2
        super(UAMFD_Net, self).__init__()

        self.text_model = TextEncoder(input_size=768, out_size=self.unified_dim, rnn=True,
                                      rnn_type='LSTM', ch=self.ch, finetune=False)
        self.image_model = ImageEncoder(input_dim=512, inter_dim=500, output_dim=self.unified_dim)
        self.text_attention = TokenAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.mm_attention = TokenAttention(self.unified_dim)
        # GATE, EXPERTS
        # feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64} # 64*5 note there are 5 kernels and 5 experts!
        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(MCO(input_size=self.unified_dim, nhead=4,
                                        dim_feedforward=2 * self.unified_dim,
                                        dropout=0.2))  # note: need to output model[:,0]

            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        for i in range(self.num_expert):
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(MCO(input_size=self.unified_dim, nhead=4,
                                       dim_feedforward=2 * self.unified_dim, dropout=0.2))
                mm_expert.append(MCO(input_size=self.unified_dim, nhead=4,
                                     dim_feedforward=2 * self.unified_dim, dropout=0.2))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        # self.out_unified_dim = 320
        self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.SiLU(),
                                            nn.Linear(self.unified_dim, self.num_expert),
                                            )

        self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     # SimpleGate(),
                                     # nn.BatchNorm1d(int(self.unified_dim/2)),
                                     nn.Linear(self.unified_dim, self.num_expert),

                                     )
        self.mm_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )

        self.image_gate_mae_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                              nn.SiLU(),
                                              # SimpleGate(),
                                              # nn.BatchNorm1d(int(self.unified_dim/2)),
                                              nn.Linear(self.unified_dim, self.num_expert),
                                              # nn.Dropout(0.1),
                                              # nn.Softmax(dim=1)
                                              )

        self.text_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )

        ## MAIN TASK GATES
        self.final_attention = TokenAttention(self.unified_dim)

        self.fusion_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                         nn.SiLU(),
                                                         # SimpleGate(),
                                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                                         nn.Linear(self.unified_dim, self.num_expert),
                                                         # nn.Softmax(dim=1)
                                                         )

        self.irrelevant_tensor = nn.Parameter(torch.ones((1, self.unified_dim)), requires_grad=True)

        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.aux_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        #### mapping MLPs
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(
                2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_mu = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_sigma = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(2, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        final_fusing_expert = []
        for i in range(self.num_expert):
            fusing_expert = []
            for j in range(self.depth):
                fusing_expert.append(MCO(input_size=self.unified_dim, nhead=4,
                                         dim_feedforward=2 * self.unified_dim, dropout=0.2))
            fusing_expert = nn.ModuleList(fusing_expert)
            final_fusing_expert.append(fusing_expert)

        self.final_fusing_experts = nn.ModuleList(final_fusing_expert)

        self.mm_score = None

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity, category=None,
                              calc_ambiguity=False):
        image_feature = self.image_model.forward_ying(image)
        text_feature = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]

        return image_feature, text_feature

    def forward(self, imgs, encoded_texts, mask_batch_T, mask_batch_TT,
                mask_batch_TV, adj_matrix, word_len, word_spans):

        # print(input_ids.shape) # (24,197)
        # print(attention_mask.shape) # (24,197)
        # print(token_type_ids.shape) # (24,197)

        text_feature = self.text_model(encoded_texts, word_spans, word_len)
        image_feature = self.image_model(imgs)

        # (N, outdim)
        text_atn_feature, _ = self.text_attention(text_feature)

        # (N, outdim)
        image_atn_feature, _ = self.image_attention(image_feature)
        # (N, outdim)
        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        # (N, num_expert)
        gate_image_feature = self.image_gate_mae(image_atn_feature)
        # (N, num_expert)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320
        # (N, num_expert)
        gate_mm_feature = self.mm_gate(mm_atn_feature)
        # (N, num_expert)
        gate_mm_feature_1 = self.mm_gate_1(mm_atn_feature)
        # image expert
        shared_image_feature, shared_image_feature_1 = 0, 0
        # replace the expert with multi-head attention
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tgt=tmp_image_feature, src=tmp_image_feature, src_key_padding_mask=None)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))
        # the original paper chooses the first token, (N,B)
        shared_image_feature = torch.mean(shared_image_feature, dim=1)

        # TEXT EXPERTS
        shared_text_feature = 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tgt=tmp_text_feature, src=tmp_text_feature, src_key_padding_mask=None) # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = torch.mean(shared_text_feature, dim=1)

        # Image Experts
        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        shared_mm_feature, shared_mm_feature_CC = 0, 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tgt=tmp_mm_feature, src=tmp_mm_feature, src_key_padding_mask=None)
            shared_mm_feature += (tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(1).unsqueeze(1))
            shared_mm_feature_CC += (tmp_mm_feature * gate_mm_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = torch.mean(shared_mm_feature, dim=1)
        shared_mm_feature_CC = torch.mean(shared_mm_feature_CC, dim=1)
        # map to low-dimension 64
        shared_mm_feature_lite = self.aux_trim(shared_mm_feature_CC)
        # aux classify
        aux_output = self.aux_classifier(shared_mm_feature_lite)  # final_feature_aux_task

        ## UNIMODAL BRANCHES, NOT USED ANY MORE
        # aux_output = aux_output.clone().detach()

        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        aux_atn_score = 1 - torch.sigmoid(
            aux_output).clone().detach()  # torch.abs((torch.sigmoid(aux_output).clone().detach()-0.5)*2)
        is_mu = self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output).clone().detach())
        t_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        cc_mu = self.mapping_CC_MLP_mu(aux_atn_score.clone().detach())  # 1-aux_atn_score
        is_sigma = self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output).clone().detach())
        t_sigma = self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output).clone().detach())
        cc_sigma = self.mapping_CC_MLP_sigma(aux_atn_score.clone().detach())  # 1-aux_atn_score

        shared_image_feature = self.adaIN(shared_image_feature, is_mu,
                                          is_sigma)  # shared_image_feature * (image_atn_score)
        shared_text_feature = self.adaIN(shared_text_feature, t_mu, t_sigma)  # shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature  # shared_mm_feature #* (aux_atn_score)
        irr_score = torch.ones_like(
            shared_mm_feature) * self.irrelevant_tensor  # torch.ones_like(shared_mm_feature).cuda()
        irrelevant_token = self.adaIN (irr_score, cc_mu, cc_sigma)
        concat_feature_main_biased = torch.stack((shared_image_feature,
                                                  shared_text_feature,
                                                  shared_mm_feature,
                                                  irrelevant_token
                                                  ), dim=1)

        fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_main_biased)
        gate_main_task = self.fusion_SE_network_main_task(fusion_tempfeat_main_task)

        final_feature_main_task = 0
        for i in range(self.num_expert):
            fusing_expert = self.final_fusing_experts[i]
            tmp_fusion_feature = concat_feature_main_biased
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](tgt=tmp_fusion_feature, src=tmp_fusion_feature, src_key_padding_mask=None)
            tmp_fusion_feature = torch.mean(tmp_fusion_feature, dim=1)
            final_feature_main_task += (tmp_fusion_feature * gate_main_task[:, i].unsqueeze(1))

        final_feature_main_task_lite = self.mix_trim(final_feature_main_task)
        mix_output = self.mix_classifier(final_feature_main_task_lite)

        return mix_output, image_only_output, text_only_output, aux_output, \
               torch.mean(self.irrelevant_tensor)

    def mapping(self, score):
        ## score is within 0-1
        diff_with_thresh = torch.abs(score - self.thresh)
        interval = torch.where(score - self.thresh > 0, 1 - self.thresh, self.thresh)
        return diff_with_thresh / interval