import json

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
import random
import numpy as np
from models import RFND, calculate_probability, Vis_RFND
import argparse
from utils import seed_everything
from utils import Twitter_Set, Weibo_Set, PadCollate, CompleteLogger, all_metrics
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import shutil
import torch.nn.functional as F
import time
from tqdm import tqdm
from comet_ml import Experiment
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')
# device 3 for 1e-3

def get_parser():
    parser = argparse.ArgumentParser(description='Multimodal TextCnn')
    # dataset parameters and log parameter
    parser.add_argument('-d', '--data', metavar='DATA', default='weibo', choices=['weibo', 'twitter'],
                        help='dataset: weibo, twitter')
    parser.add_argument('--maxlength', type=int, default=150)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--tag", type=str, default="LogicDM",
                        help="the tags for comet")
    parser.add_argument('--seed', default=0
                        , type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--type', default='fix'
                        , type=str,
                        help='seed for initializing training. ')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--patience', default=5, type=int, metavar='M',
                        help='patience')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')


    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument("--log", type=str, default='//data/sunhao/rulenews/final-model-58444',
                        help="Where to save logs, checkpoints and debugging images.")
    # //data/sunhao/rulenews/sarcasm-final-68713
    # //data/sunhao/rulenews/twitter-final-model-29354
    # //data/sunhao/rulenews/final-model-58444
    # model parameter
    parser.add_argument("--graphtype", type=str, default='cross',
                        help="the type of cross modal graph")
    parser.add_argument('--outsize', default=64, type=int)
    parser.add_argument('--sizeclues', default='10#10#10#10#10', type=str)
    parser.add_argument('--rnntype', default='LSTM', type=str, choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--instancedim', default=64, type=int)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--finetune', default='false', type=str)
    parser.add_argument('--relationdim', default=64, type=int)
    parser.add_argument('--ansnumber', default=2, type=int)
    parser.add_argument('--answerdim', default=64, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--rate', default=0.1, type=float)
    parser.add_argument('--normtype', default='product', type=str, choices=['minimum',  'product', 'lukasiewicz'])
    parser.add_argument('--rnn', default='true', type=str)
    parser.add_argument('--hop', default='three', type=str, choices=['one', 'two', 'three', 'twotothree', 'onetothree', 'onetotwo'])
    parser.add_argument('--loss-type', default='false', type=str)
    parser.add_argument('--guidehead', default=5, type=int)
    #  graph layer + 1
    parser.add_argument('--gcnumber', default=3, type=int)

    parser.add_argument('--max_iter', default=30, type=int,
                        help='the maximum number of iteration in each epoch ')
    parser.add_argument('--d_rop', default=0.3, type=float,
                        help='d_rop rate of neural network model')

    args = parser.parse_args()
    if args.finetune in ["True", "true"]:
        args.finetune = True
    else:
        args.finetune = False
    if args.rnn  in ['true', 'True']:
        args.rnn = True
    else:
        args.rnn = False

    args.sizeclues = [int(i) for i in str(args.sizeclues).strip().split('#')]
    return args


def main(args, experiment=None):
    logger = CompleteLogger(args.log, args.phase)
    # logger = None

    if args.data == 'twitter':
        if args.type == 'random':
            dataset = Twitter_Set(max_length=args.maxlength)
            train_size = int(len(dataset) * args.split)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        else:
            train_dataset = Twitter_Set(max_length=args.maxlength, mode='fix', phase='train')
            test_dataset = Twitter_Set(max_length=args.maxlength, mode='fix', phase='test')

        ch = False
    elif args.data == 'weibo':
        if args.type == 'random':
            dataset = Weibo_Set(max_length=args.maxlength)
            train_size = int(len(dataset) * args.split)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        else:
            train_dataset = Weibo_Set(max_length=args.maxlength, mode='fix', phase='train')
            test_dataset = Weibo_Set(max_length=args.maxlength, mode='fix', phase='test')
        ch = True

    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers,
                                     shuffle=False)

    RFND_Model = Vis_RFND(input_size=768, out_size=args.outsize, rnn=args.rnn, rnn_type=args.rnntype, ch=ch, finetune=args.finetune,
                 instance_dim=args.instancedim, top_K=args.topk, size_clues=args.sizeclues, relation_dim=args.relationdim, ans_number=args.ansnumber, answer_dim=args.answerdim,
                 guide_head=args.guidehead, norm_type=args.normtype, threshold=args.threshold, rate=args.rate, gcnumber=args.gcnumber)

    RFND_Model.to(device=device)
    parameters = RFND_Model.parameters()

    print("all parameters", sum(p.numel() for p in RFND_Model.parameters() if p.requires_grad))
    print("bert parameters", sum(p.numel() for p in RFND_Model.text_encoder.bert_model.parameters() if p.requires_grad))


    optimizer = optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.wd,
                           amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)

    # if args.phase != 'train':
    #     checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    #     RFND_Model.load_state_dict(checkpoint)

    """
    acc: the number of samples that is accurately predicted / the number of samples
    precision: The percentage of examples classified as positive examples that are actually positive examples.
    recall:  the percentage of positive cases that are correct,
    F: 2*P*R/(P+R)
    """

    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        RFND_Model.load_state_dict(checkpoint)
        test_loss, metrics_test = validate(test_loader, RFND_Model, args, device)
        return test_loss, metrics_test

    # start training
    best_acc1 = 0.
    acc1_store = []
    best_metric = None
    prefix = "seed-" + str(args.seed)
    for epoch in range(args.epochs):
        # train for one epoch
        # evaluate on validation set
        train_loss, metrics_train = train_model(dataloader=train_loader, model=RFND_Model, optimizer=optimizer,
                                                epoch=epoch, args=args)
        # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
        test_loss, metrics_test = validate(test_loader, RFND_Model, args=args, device=device)
        train_metircs = {prefix + "-train_acc": metrics_train[0], prefix + "-train_loss": float(train_loss),
                          prefix + "-train_r_romor": metrics_train[1], prefix + "-train_p_rumor": metrics_train[2],
                          prefix + "-train_f1_romor": metrics_train[3], prefix + "-train_r_non_rumor": metrics_train[4],
                          prefix + "-train_p_non_romor": metrics_train[5], prefix + "-train_f1_non_rumor": metrics_train[6]
                          }
        test_metircs = {prefix + "-test_acc": metrics_test[0], prefix + "-test_loss": float(test_loss),
                          prefix + "-test_r_romor": metrics_test[1], prefix + "-test_p_rumor": metrics_test[2],
                          prefix + "-test_f1_romor": metrics_test[3], prefix + "-test_r_non_rumor": metrics_test[4],
                          prefix + "-test_p_non_romor": metrics_test[5], prefix + "-test_f1_non_rumor": metrics_test[6]
                          }


        # f1_store.append(float(f1_target))
        # acc1_store.append(metrics_test[0])
        lr_scheduler.step(float(metrics_test[0]))
        # remember best acc@1 and save checkpoint
        torch.save(RFND_Model.state_dict(), logger.get_checkpoint_path('latest'))
        if metrics_test[0] > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_metric = metrics_test
        best_acc1 = max(metrics_test[0], best_acc1)
    print("best_acc1 = {:.4f}".format(best_acc1))

    return best_metric


def validate(dataloader, model, args, device):
    model.eval()
    end = time.time()
    real_labels = []
    predicted_labels = []
    # final_test_loss = 0
    final_test_loss = 0

    all_data_gc_E_5k = []
    all_gc_mh_head_logic_score = []
    all_gc_mh_head_logic_clauses =[]
    all_gc_mh_head_index = []
    all_final_logic_score = []
    all_predicate_label = []
    all_word_len_list = []
    for i, (imgs, encoded_texts,  mask_batch_text,
               adj_matrix, word_len, mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}

        mask_batch_text = mask_batch_text.cuda()
        adj_matrix = adj_matrix.cuda()
        mask_T_T = mask_T_T.cuda()
        mask_T_V = mask_T_V.cuda()
        labels = labels.cuda()
        all_word_len_list = all_word_len_list + word_len
        with torch.no_grad():
            gc_clause_list, gc_E_5k, gc_mh_head_logic_score, gc_mh_head_logic_clauses, gc_mh_head_index = model(imgs=imgs, encoded_texts=encoded_texts,
                                   mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                                   mask_batch_TV=mask_T_V,  adj_matrix=adj_matrix,
                                   word_len=word_len,  word_spans=word_spans)
            # N, 2
            probability = calculate_probability(gc_clause_list, args.hop, args.normtype)
            all_data_gc_E_5k.append(gc_E_5k)
            all_gc_mh_head_logic_score.append(gc_mh_head_logic_score)
            all_gc_mh_head_logic_clauses.append(gc_mh_head_logic_clauses)
            all_gc_mh_head_index.append(gc_mh_head_index)

            final_test_loss = final_test_loss + F.cross_entropy(probability, labels).item()
            real_labels = real_labels + labels.cpu().detach().clone().numpy().tolist()
            # the [1] is fake
            predicted_labels = predicted_labels + (probability[:, 0] < probability[:, 1]).cpu().long().numpy().tolist()
            all_final_logic_score.append(probability.clone().detach().cpu())
            all_predicate_label = all_predicate_label + predicted_labels
            torch.cuda.empty_cache()
    period = time.time() - end
    metrics_test = all_metrics(real_labels, predicted_labels)
    """
    gc_E_5k: 3, 32, 25
    gc_mh_head_logic_score: 3, 5, 2, 32, 2 (3 gcn,5 head,2 answer, 32 batchsize, 2 length of clause)
    gc_mh_head_logic_clauses: 3, 5, 2, 32, 2 (3 gcn,5 head,2 answer, 32 batchsize, 2 length of clause)
    gc_mh_head_index: 3, 2, 32
    """
    all_data_gc_E_5k = torch.cat(all_data_gc_E_5k, dim=1)
    all_gc_mh_head_logic_score = torch.cat(all_gc_mh_head_logic_score, dim=-2)
    all_gc_mh_head_logic_clauses = torch.cat(all_gc_mh_head_logic_clauses, dim=-2)
    all_gc_mh_head_index = torch.cat(all_gc_mh_head_index, dim=-1)
    all_final_logic_score = torch.cat(all_final_logic_score, dim=0)
    all_word_len_list = torch.tensor(all_word_len_list)

    torch.save(all_data_gc_E_5k, os.path.join(args.log, "test_all_data_gc_E_5k.pt"))
    torch.save(all_gc_mh_head_logic_score, os.path.join(args.log, "test_all_gc_mh_head_logic_score.pt"))
    torch.save(all_gc_mh_head_logic_clauses, os.path.join(args.log, "test_all_gc_mh_head_logic_clauses.pt"))
    torch.save(all_gc_mh_head_index, os.path.join(args.log, "test_all_gc_mh_head_index.pt"))
    torch.save(all_final_logic_score, os.path.join(args.log, "test_all_final_logic_score.pt"))
    torch.save(all_word_len_list, os.path.join(args.log, "test_all_word_len.pt"))

    with open(os.path.join(args.log, "test_all_predicate_label.json"), 'w')as f:
        json.dump(all_predicate_label, f)


    # torch.save(all_data_gc_E_5k, os.path.join(args.log, "train_all_data_gc_E_5k.pt"))
    # torch.save(all_gc_mh_head_logic_score, os.path.join(args.log, "train_all_gc_mh_head_logic_score.pt"))
    # torch.save(all_gc_mh_head_logic_clauses, os.path.join(args.log, "train_all_gc_mh_head_logic_clauses.pt"))
    # torch.save(all_gc_mh_head_index, os.path.join(args.log, "train_all_gc_mh_head_index.pt"))
    # torch.save(all_final_logic_score, os.path.join(args.log, "train_all_final_logic_score.pt"))
    # torch.save(all_word_len_list, os.path.join(args.log, "train_all_word_len.pt"))
    #
    # with open(os.path.join(args.log, "train_all_predicate_label.json"), 'w')as f:
    #     json.dump(all_predicate_label, f)


    # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
    print("Test: Time: {:.4f}, Acc: {:.4f}, Loss: {:.4f}, Rumor_R: {:.4f}, Rumor_P: {:.4f}, "
          "Rumor_F: {:.4f}, Non_Rumor_R: {:.4f}, Non_Rumor_P: {:.4f}, Non_Rumor_F1: {:.4f}".format(period,
                                                                                               metrics_test[0],
                                                                                               final_test_loss/(len(dataloader)),
                                                                                               metrics_test[1],
                                                                                               metrics_test[2],
                                                                                               metrics_test[3],
                                                                                               metrics_test[4],
                                                                                               metrics_test[5],
                                                                                               metrics_test[6]))
    return final_test_loss, metrics_test


def train_model(dataloader, model, optimizer, epoch, args):
    model.train()
    final_train_loss = 0
    end = time.time()
    real_labels  = []
    predicted_labels = []
    for i, (imgs, encoded_texts, mask_batch_text,
               adj_matrix, word_len,  mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
        mask_batch_text = mask_batch_text.cuda()
        adj_matrix = adj_matrix.cuda()
        mask_T_T = mask_T_T.cuda()
        mask_T_V = mask_T_V.cuda()
        labels = labels.cuda()
        # Gc_number_layer, 2, N;
        gc_clause_list, gc_E_5k, gc_mh_head_logic_score, gc_mh_head_logic_clauses, gc_mh_head_index  = model(imgs=imgs, encoded_texts=encoded_texts,  mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                mask_batch_TV=mask_T_V, adj_matrix=adj_matrix,
                               word_len=word_len,  word_spans=word_spans)
        # N, 2
        probability= calculate_probability(gc_clause_list, args.hop, args.normtype)
        # train_loss = F.cross_entropy(probability, labels)
        # introduce another kind of loss
        train_loss = F.cross_entropy(probability, labels) +\
                     F.cross_entropy(torch.stack([probability[:, 0], (1-probability)[:, 0]], dim=1), labels) + \
                     F.cross_entropy(torch.stack([(1 - probability)[:, 1], probability[:, 1]], dim=1), labels)


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        final_train_loss = final_train_loss + train_loss.item()
        real_labels = real_labels + labels.cpu().detach().clone().numpy().tolist()
        # the [1] is score
        predicted_labels = predicted_labels + (probability[:, 0] < probability[:, 1]).cpu().long().numpy().tolist()
        torch.cuda.empty_cache()
    period = time.time() - end
    metrics_train = all_metrics(real_labels, predicted_labels)
        # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
    print("Train Epoch {}: Time {:.4f}, Acc: {:.4f}, Loss: {:.4f}, Rumor_R: {:.4f}, Rumor_P: {:.4f}, "
              "Rumor_F: {:.4f}, Non_Rumor_R: {:.4f}, Non_Rumor_P: {:.4f}, Non_Rumor_F1: {:.4f}".format(epoch, period,
        metrics_train[0], final_train_loss/(len(dataloader)), metrics_train[1],  metrics_train[2], metrics_train[3],
        metrics_train[4],  metrics_train[5], metrics_train[6]))

    return final_train_loss, metrics_train


if __name__ == '__main__':
    # must record every result
    args = get_parser()
    experiment = None

    metrics_train_final = main(args, experiment=experiment)


