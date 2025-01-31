import sys
import logging
from code_generation.exception import CodeGeneratorException
import torch.nn.functional as F
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True) -> None:
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        try:
            if smooth_dist is None:
                smooth_dist = self.smooth_dist
            return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                                reduction=self.reduction, smooth_eps=self.smooth_eps,
                                smooth_dist=smooth_dist, from_logits=self.from_logits)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    try:
        """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
        smooth_eps = smooth_eps or 0
        # ordinary log-liklihood - use cross_entropy from nn
        if _is_long(target) and smooth_eps == 0:
            if from_logits:
                return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
            else:
                return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

        if from_logits:
            # log-softmax of inputs
            lsm = F.log_softmax(inputs, dim=-1)
        else:
            lsm = inputs

        masked_indices = None
        num_classes = inputs.size(-1)

        if _is_long(target) and ignore_index >= 0:
            masked_indices = target.eq(ignore_index)

        if smooth_eps > 0 and smooth_dist is not None:
            if _is_long(target):
                target = onehot(target, num_classes).type_as(inputs)
            if smooth_dist.dim() < target.dim():
                smooth_dist = smooth_dist.unsqueeze(0)
            target.lerp_(smooth_dist, smooth_eps)

        if weight is not None:
            lsm = lsm * weight.unsqueeze(0)

        if _is_long(target):
            eps_sum = smooth_eps / num_classes
            eps_nll = 1. - eps_sum - smooth_eps
            likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
        else:
            loss = -(target * lsm).sum(-1)

        if masked_indices is not None:
            loss.masked_fill_(masked_indices, 0)

        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            if masked_indices is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

        return loss

    except Exception as e:
        raise CodeGeneratorException(e, sys) from e


def onehot(indexes, N=None, ignore_index=None):
    try:
        """
        Creates a one-representation of indexes with N possible entries
        if N is not specified, it will suit the maximum index appearing.
        indexes is a long-tensor of indexes
        ignore_index will be zero in onehot representation
        """
        if N is None:
            N = indexes.max() + 1
        sz = list(indexes.size())
        output = indexes.new().byte().resize_(*sz, N).zero_()
        output.scatter_(-1, indexes.unsqueeze(-1), 1)
        if ignore_index is not None and ignore_index >= 0:
            output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
        return output

    except Exception as e:
        raise CodeGeneratorException(e, sys) from e
        

def _is_long(x):
    try:
        if hasattr(x, 'data'):
            x = x.data
        return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)

    except Exception as e:
        raise CodeGeneratorException(e, sys) from e