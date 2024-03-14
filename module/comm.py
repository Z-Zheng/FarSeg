import ever.module.loss as L
import numpy as np
import torch
import torch.nn.functional as F

from ever.core.dist import get_world_size
import torch.distributed as dist
import math


def all_reduce_sum(data):
    if get_world_size() == 1:
        return data
    dist.all_reduce(data)


def _iou_1(y_true, y_pred, ignore_index=None):
    with torch.no_grad():
        if ignore_index:
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            valid = y_true != ignore_index
            y_true = y_true.masked_select(valid).float()
            y_pred = y_pred.masked_select(valid).float()
        y_pred = y_pred.float().reshape(-1)
        y_true = y_true.float().reshape(-1)
        inter = torch.sum(y_pred * y_true)
        union = y_true.sum() + y_pred.sum()
        return inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))


@torch.jit.script
def _softmax_focal_loss(y_pred, y_true, ignore_index: int = 255, gamma: float = 2.0):
    ce_losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    # with torch.no_grad():
    p = y_pred.softmax(dim=1)
    modulating_factor = (1 - p).pow(gamma)
    valid_mask = ~ y_true.eq(ignore_index)
    masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
    modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
    foc_losses = ce_losses * modulating_factor
    return ce_losses, foc_losses, valid_mask


def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)


def annealing_softmax_focal_loss(y_pred,
                                 y_true,
                                 t: float,
                                 t_max: float,
                                 ignore_index: int = 255,
                                 gamma: float = 2.0,
                                 normalize: bool = False,
                                 ):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    EPS: float = 1e-7
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
            if t > t_max:
                scale = scale
            else:
                scale = cosine_annealing(1., scale, t, t_max)

    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + EPS)
    return losses


def sync_annealing_softmax_focal_loss(y_pred,
                                      y_true,
                                      t: float,
                                      t_max: float,
                                      ignore_index: int = 255,
                                      gamma: float = 2.0,
                                      normalize: bool = False,
                                      ):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    EPS: float = 1e-7

    ce_losses, foc_losses, valid_mask = _softmax_focal_loss(y_pred, y_true, ignore_index, gamma)

    if normalize:
        ce_sum = ce_losses.sum()
        foc_sum = foc_losses.sum()
        ce_foc_sum = torch.stack([ce_sum, foc_sum], dim=0)
        all_reduce_sum(ce_foc_sum)

        scale = ce_foc_sum[0] / ce_foc_sum[1]

        if t > t_max:
            scale = scale
        else:
            scale = cosine_annealing(1., scale, t, t_max)

    losses = scale * foc_losses.sum() / (valid_mask.sum() + EPS)
    return losses


class MultiSegmentation(object):
    def loss(self, y_true: torch.Tensor, y_pred, loss_config, **kwargs):
        loss_dict = dict()

        if 'prefix' in loss_config:
            prefix = loss_config.prefix
        else:
            prefix = ''

        if 'mem' in loss_config:
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)

        if 'bce' in loss_config:
            weight = loss_config.bce.get('weight', 1.0)
            loss_dict[f'{prefix}bce@w{weight}_loss'] = weight * L.binary_cross_entropy_with_logits(y_pred, y_true,
                                                                                                   reduction='mean',
                                                                                                   ignore_index=loss_config.ignore_index)
            del weight

        if 'ce' in loss_config:
            weight = loss_config.ce.get('weight', 1.0)
            loss_dict[f'{prefix}ce@w{weight}_loss'] = weight * F.cross_entropy(y_pred, y_true.long(),
                                                                               ignore_index=loss_config.ignore_index)
            del weight

        if 'annealing_softmax_focal' in loss_config:
            weight = loss_config.annealing_softmax_focal.get('weight', 1.0)
            is_normalize = loss_config.annealing_softmax_focal.get('normalize', False)
            t_max = loss_config.annealing_softmax_focal.t_max
            if loss_config.annealing_softmax_focal.get('sync', False):
                loss_dict[f'{prefix}syncfocal@w{weight}_loss'] = weight * sync_annealing_softmax_focal_loss(y_pred,
                                                                                                            y_true.long(),
                                                                                                            t=kwargs[
                                                                                                                'buffer_step'],
                                                                                                            t_max=t_max,
                                                                                                            ignore_index=loss_config.ignore_index,
                                                                                                            normalize=is_normalize)
            else:
                loss_dict[f'{prefix}focal@w{weight}_loss'] = weight * annealing_softmax_focal_loss(y_pred,
                                                                                                   y_true.long(),
                                                                                                   t=kwargs['buffer_step'],
                                                                                                   t_max=t_max,
                                                                                                   ignore_index=loss_config.ignore_index,
                                                                                                   normalize=is_normalize)

        if 'dice' in loss_config:
            ignore_channel = loss_config.dice.get('ignore_channel', -1)
            weight = loss_config.dice.get('weight', 1.0)
            loss_dict[f'{prefix}dice@w{weight}_loss'] = weight * L.dice_loss_with_logits(y_pred, y_true,
                                                                                         ignore_index=loss_config.ignore_index,
                                                                                         ignore_channel=ignore_channel)
            del weight

        if 'log_objectness_iou_sigmoid' in loss_config:
            with torch.no_grad():
                _y_pred, _y_true = L.select(y_pred, y_true, loss_config.ignore_index)
                _binary_y_true = (_y_true > 0).float()
                cls = (_y_pred.sigmoid() > 0.5).float()

            loss_dict[f'{prefix}obj_iou'] = _iou_1(_binary_y_true, cls)

        if 'log_objectness_iou' in loss_config:
            with torch.no_grad():
                _y_pred, _y_true = L.select(y_pred, y_true, loss_config.ignore_index)
                _binary_y_true = (_y_true > 0).float()
                cls = (_y_pred.argmax(dim=1) > 0).float()

            loss_dict[f'{prefix}obj_iou'] = _iou_1(_binary_y_true, cls)

        return loss_dict
