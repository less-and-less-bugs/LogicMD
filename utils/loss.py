import torch

device = 'cuda'
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

    neg_predictions = torch.cat([predictions[:label], predictions[label+1:]], dim=0)
    pos_prediction = predictions[label]
    loss = torch.mean(torch.max(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device) + neg_predictions - pos_prediction))

    return loss# + c_entropy * entropy