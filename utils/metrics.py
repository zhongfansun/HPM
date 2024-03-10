import torch


def soft_acc(logits, labels):
    """ estimate acc according to VQA definition
    Args:
        logits (tensor): shape of (b, num_answer)
        labels (tensor): shape of (b, num_answer)

    """
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores


def top_acc(logits, labels):
    """ estimate top1 and top3 acc
    Args:
        logits (tensor): shape of (b, num_answer)
        labels (tensor): shape of (b, num_answer)
    """
    top1, top3 = [], []
    logits = logits.topk(k=3, dim=-1)[1]
    labels = labels.max(dim=-1)[1]
    for logit, label in zip(logits, labels):
        if label == logit[0]:
            top1.append(1.0)
        else: 
            top1.append(0.0)
            
        if label in logit:
            top3.append(1.0)
        else:
            top3.append(0.0)
    return torch.tensor(top1), torch.tensor(top3)
