import torch
import torch.nn.functional as F
import math


def softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0, normalize=False):
    """

    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar

    Returns:

    """
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses


def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)


def poly_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = (1 - _t / _t_max) ** 0.9
    return upper_bound + factor * (lower_bound - upper_bound)


def linear_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = 1 - _t / _t_max
    return upper_bound + factor * (lower_bound - upper_bound)


def annealing_softmax_focalloss(y_pred, y_true, t, t_max, ignore_index=255, gamma=2.0,
                                annealing_function=cosine_annealing):
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        normalizer = losses.sum() / (losses * modulating_factor).sum()
        scales = modulating_factor * normalizer
    if t > t_max:
        scale = scales
    else:
        scale = annealing_function(1, scales, t, t_max)
    losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
    return losses
