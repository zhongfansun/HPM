from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR


def CE_loss(logits, labels, weights):
    """ Modified cross entropy loss. """
    nll = F.log_softmax(logits, dim=-1)
    loss = - (nll * labels).sum(dim=-1)
    if weights is not None:
        loss = loss * weights
    return loss


class PlainLoss(nn.Module):
    def __init__(self, loss_type) -> None:
        super().__init__()
        self.loss_type = loss_type

    def forward(self, logits, labels, weights=None):
        if self.loss_type == 'ce':
            loss = CE_loss(logits, labels, weights)
        else: # bce
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = loss.mean() if weights is None else (loss * weights).mean()
            loss *= labels.size(1)
        return loss


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0. to 1. 
        over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. 
        over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))
        