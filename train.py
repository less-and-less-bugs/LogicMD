import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
# from tqdm import tqdm
# from dataset import FeatureDataset
from model import SimilarityModule, DetectionModule


# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 100


def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image


def train():
    # ---  Load Config  ---

    


    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule()  
    detection_module = DetectionModule()

    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    # ---  Model Training  ---
    loss_similarity_total = 0
    loss_detection_total = 0
    best_acc = 0


def test(similarity_module, detection_module, test_loader):
    similarity_module.eval()
    detection_module.eval()

    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()

    similarity_count = 0
    detection_count = 0
    loss_similarity_total = 0
    loss_detection_total = 0
    similarity_label_all = []
    detection_label_all = []
    similarity_pre_label_all = []
    detection_pre_label_all = []

    with torch.no_grad():
        for i, (text, image, label) in enumerate(test_loader):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)
            
            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            # ---  TASK1 Similarity  ---

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text, matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text, unmatched_image)
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            # ---  TASK2 Detection  ---

            text_aligned, image_aligned, _ = similarity_module(text, image)
            pre_detection = detection_module(text, image, text_aligned, image_aligned)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---

            loss_similarity_total += loss_similarity.item() * (2 * fixed_text.shape[0])
            loss_detection_total += loss_detection.item() * text.shape[0]
            similarity_count += (fixed_text.shape[0] * 2)
            detection_count += text.shape[0]

            similarity_pre_label_all.append(similarity_pred.detach().cpu().numpy())
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            similarity_label_all.append(similarity_label_0.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        loss_similarity_test = loss_similarity_total / similarity_count
        loss_detection_test = loss_detection_total / detection_count

        similarity_pre_label_all = np.concatenate(similarity_pre_label_all, 0)
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        similarity_label_all = np.concatenate(similarity_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_similarity_test = accuracy_score(similarity_pre_label_all, similarity_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_similarity = confusion_matrix(similarity_pre_label_all, similarity_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)

    return acc_similarity_test, acc_detection_test, loss_similarity_test, loss_detection_test, cm_similarity, cm_detection


if __name__ == "__main__":
    similarity_module = SimilarityModule()
    detection_module = DetectionModule()
    print("parameters of similarity_module", sum(p.numel() for p in similarity_module.parameters() if p.requires_grad))
    print("parameters of detection_module", sum(p.numel() for p in detection_module.parameters() if p.requires_grad))



