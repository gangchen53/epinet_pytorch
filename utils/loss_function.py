from typing import Optional

import torch
import torch.nn as nn


def create_mask_margin(shape, margin: int):
    """ Create a mask with a False margin. """
    assert margin >= 0

    mask = torch.ones(shape, dtype=torch.bool)

    if margin > 0:
        mask[..., :margin, :] = False
        mask[..., -margin:, :] = False
        mask[..., :margin] = False
        mask[..., -margin:] = False

    return mask


class LossFunction(nn.Module):
    def __init__(self, loss_type: str = 'gaussian', demarcation_point: Optional[float] = None):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.demarcation_point = demarcation_point

    def __repr__(self):
        if self.loss_type == 'mae':
            return 'L1 Loss'
        else:
            raise NotImplementedError

    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                variance: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ):
        if self.loss_type == 'gaussian':
            assert variance, 'if loss function is gaussian, variance must not be None!'

            mean = predictions

            variance[variance < 1e-10] = 1e-10

            loss = 1 / 2 * torch.log(variance) + (labels - mean) ** 2 / (2 * variance)
            loss = torch.mean(loss)
            return loss
        elif self.loss_type == 'smooth gaussian':
            assert variance, 'if loss function is smooth gaussian, variance must not be None!'
            assert self.demarcation_point, 'if loss function is smooth gaussian, demarcation_point must be not None!'

            mean = predictions

            variance[variance < 1e-10] = 1e-10

            diff = torch.abs(labels - mean)
            mask = diff < self.demarcation_point

            # loss for Gaussian distribution
            loss_1 = 1 / 2 * torch.log(variance[mask]) + (labels[mask] - mean[mask]) ** 2 / (2 * variance[mask])

            # loss for Laplace distribution
            loss_2 = 1 / 2 * torch.log(torch.sqrt(variance[~mask])) + torch.abs(labels[~mask] - mean[~mask]) / (
                    2 * torch.sqrt(variance[~mask]))

            loss = torch.cat((loss_1, loss_2), dim=0)
            loss = torch.mean(loss)
            return loss
        elif self.loss_type == 'mae':
            diff = torch.abs(torch.flatten(predictions) - torch.flatten(labels))

            if mask is not None:
                count = mask.int().sum()
                diff *= torch.flatten(mask).float()

                loss = diff.sum() / count
            else:
                loss = diff.sum() / diff.size(0)

            # loss = torch.nn.L1Loss()(predictions, labels)
            return loss
        else:
            raise NotImplementedError
