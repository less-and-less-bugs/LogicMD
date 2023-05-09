import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
import random
import numpy as np
from models import RFND, calculate_probability
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')
# device 3 for 1e-3

def get_parser():
    parser = argparse.ArgumentParser(description='Multimodal TextCnn')
    # dataset parameters and log parameter
    parser.add_argument('-d', '--data', metavar='DATA', default='twitter', choices=['weibo', 'twitter'],
                        help='dataset: weibo, twitter')
    parser.add_argument('--maxlength', type=int, default=150)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--tag", type=str, default="roberta-non",
                        help="the tags for comet")
    parser.add_argument('--seed', default=0
                        , type=int,
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
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')


    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument("--log", type=str, default='jan',
                        help="Where to save logs, checkpoints and debugging images.")

    # model parameter
    parser.add_argument("--graphtype", type=str, default='cross',
                        help="the type of cross modal graph")
    parser.add_argument('--outsize', default=200, type=int)
    parser.add_argument('--sizeclues', default='10#10#10#10#10', type=str)
    parser.add_argument('--rnntype', default='LSTM', type=str, choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--instancedim', default=200, type=int)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--finetune', default='false', type=str)
    parser.add_argument('--relationdim', default=200, type=int)
    parser.add_argument('--ansnumber', default=2, type=int)
    parser.add_argument('--answerdim', default=200, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--rate', default=0.1, type=float)
    parser.add_argument('--normtype', default='product', type=str, choices=['minimum',  'product', 'lukasiewicz'])
    parser.add_argument('--rnn', default='true', type=str)
    parser.add_argument('--hop', default='three', type=str, choices=['one', 'two', 'three', 'onetwo', 'onetothree'])
    parser.add_argument('--loss-type', default='false', type=str)
    parser.add_argument('--guidehead', default=5, type=int)
    #  graph layer + 1
    parser.add_argument('--gcnumber', default=3, type=int)

    parser.add_argument('--max_iter', default=30, type=int,
                        help='the maximum number of iteration in each epoch ')
    parser.add_argument('--gat', default="false", type=str,
                        help='Whether to use modality gat mechanism')
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
    dataset = None
    if args.data == 'twitter':
        dataset = Twitter_Set(max_length=args.maxlength)
        ch = False
    elif args.data == 'weibo':
        dataset = Weibo_Set(max_length=args.maxlength)
        ch = True
    if dataset is None:
        ch = None
        print("error dataset parameter")
        exit(0)
    # must after seed_everything
    train_size = int(len(dataset) * args.split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers,
                                     shuffle=False)

    RFND_Model = RFND(input_size=768, out_size=args.outsize, rnn=args.rnn, rnn_type=args.rnntype, ch=ch, finetune=args.finetune,
                 instance_dim=args.instancedim, top_K=args.topk, size_clues=args.sizeclues, relation_dim=args.relationdim, ans_number=args.ansnumber, answer_dim=args.answerdim,
                 guide_head=args.guidehead, norm_type=args.normtype, threshold=args.threshold, rate=args.rate, gcnumber=args.gcnumber)

    RFND_Model.to(device=device)
    parameters = RFND_Model.parameters()
    optimizer = optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.wd,
                           amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        RFND_Model.load_state_dict(checkpoint)

    """
    acc: the number of samples that is accurately predicted / the number of samples
    precision: The percentage of examples classified as positive examples that are actually positive examples.
    recall:  the percentage of positive cases that are correct,
    F: 2*P*R/(P+R)
    """

    if args.phase == 'test':
        test_loss, metrics_test = validate(test_loader, RFND_Model, args, device)
        return test_loss, metrics_test

    # start training
    best_acc1 = 0.
    acc1_store = []
    best_metric = []

    for epoch in range(args.epochs):
        # train for one epoch
        # evaluate on validation set
        train_loss, metrics_train = train_model(dataloader=train_loader, model=RFND_Model, optimizer=optimizer,
                                                epoch=epoch, args=args)
        # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
        test_loss, metrics_test = validate(test_loader, RFND_Model, args=args, device=device)
        # target_metircs = {prefix + "-target_acc": acc1, prefix + "-target_loss": cls_loss_target,
        #                   prefix + "-target_recall": recall_target, prefix + "-target_pre": pre_target}
        # source_metircs = {prefix + "-source_acc": cls_source, prefix + "-source_cls_loss": cls_losses_source,
        #                   prefix+"-source_inter_loss": domain_dis,  prefix+"-source_intra_loss": intra_dis,
        #                   prefix+"-source_loss": all_loss}
        # experiment.log_metrics(target_metircs, epoch=epoch)
        # experiment.log_metrics(source_metircs, epoch=epoch)

        # f1_store.append(float(f1_target))
        acc1_store.append(metrics_test[0])
        lr_scheduler.step(float(metrics_test[0]))
        # lr_scheduler_jmmd.step(float(acc1))
        # remember best acc@1 and save checkpoint
        torch.save(RFND_Model.state_dict(), logger.get_checkpoint_path('latest'))
        if metrics_test[0] > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_metric = metrics_test
        best_acc1 = max(metrics_test[0], best_acc1)
    print("best_acc1 = {:.4f}".format(best_acc1))

    logger.close()
    return best_metric


def validate(dataloader, model, args, device):
    model.eval()
    end = time.time()
    real_labels = []
    predicted_labels = []
    # final_test_loss = 0
    final_test_loss = 0
    for i, (imgs, encoded_texts, encoded_captions, mask_batch_text, mask_batch_caption,
               adj_matrix, word_len, cap_word_lens, mask_T_T, mask_T_V, word_spans, cap_word_spans, labels) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            imgs = imgs.to(device)
            encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
            # encoded_captions = {k: v.to(device) for k, v in encoded_captions.items()}
            mask_batch_text = mask_batch_text.cuda()
            # mask_batch_caption = mask_batch_caption.cuda()
            mask_batch_caption = None
            encoded_captions = None
            adj_matrix = adj_matrix.cuda()
            # word_len = word_len.cuda()
            # cap_word_lens = cap_word_lens.cuda()
            mask_T_T = mask_T_T.cuda()
            mask_T_V = mask_T_V.cuda()
            labels = labels.cuda()
            gc_clause_list = model(imgs=imgs, encoded_texts=encoded_texts, encoded_captions=encoded_captions,
                                   mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                                   mask_batch_TV=mask_T_V, mask_batch_caption=mask_batch_caption, adj_matrix=adj_matrix,
                                   word_len=word_len, cap_word_lens=cap_word_lens, word_spans=word_spans, cap_word_spans=cap_word_spans)
            # N, 2
            probability = calculate_probability(gc_clause_list, args.hop, args.normtype)
            final_test_loss = final_test_loss + F.cross_entropy(probability, labels).item()
            real_labels = real_labels + labels.cpu().detach().clone().numpy().tolist()
            # the [1] is score
            predicted_labels = predicted_labels + (probability[:, 0] < probability[:, 1]).cpu().long().numpy().tolist()
            torch.cuda.empty_cache()
    period = time.time() - end
    metrics_test = all_metrics(real_labels, predicted_labels)
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
    for i, (imgs, encoded_texts, encoded_captions, mask_batch_text, mask_batch_caption,
               adj_matrix, word_len, cap_word_lens, mask_T_T, mask_T_V, word_spans, cap_word_spans, labels) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
        encoded_captions = {k: v.to(device) for k, v in encoded_captions.items()}
        mask_batch_text = mask_batch_text.cuda()
        mask_batch_caption = mask_batch_caption.cuda()
        adj_matrix = adj_matrix.cuda()
        # word_len = word_len
        # cap_word_lens = cap_word_lens
        mask_T_T = mask_T_T.cuda()
        mask_T_V = mask_T_V.cuda()
        labels = labels.cuda()
        # Gc_number_layer, 2, N;
        gc_clause_list = model(imgs=imgs, encoded_texts=encoded_texts, encoded_captions=encoded_captions, mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                mask_batch_TV=mask_T_V, mask_batch_caption=mask_batch_caption, adj_matrix=adj_matrix,
                               word_len=word_len, cap_word_lens=cap_word_lens, word_spans=word_spans, cap_word_spans=cap_word_spans)
        # N, 2
        probability = calculate_probability(gc_clause_list, args.hop, args.normtype)
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
    seed = [0, 42, 1024]
    args.seed = random.randint(1, 10000)
    acc_seeds = []
    f1_seeds = []
    acc_rumor_seeds = []
    p_rumor_seeds = []
    r_rumor_seeds = []
    f1_rumor_seeds = []
    p_non_rumor_seeds = []
    r_non_rumor_seeds = []
    f1_non_rumor_seeds = []

    seed_everything(seed=args.seed)
    ori_log = args.log
    args.log = "-".join([ori_log, str(args.seed)])
    train_loss, metrics_train = main(args, experiment=None)
