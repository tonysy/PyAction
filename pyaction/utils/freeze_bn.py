"""
FOR DEBUG ONLY
--clrrrr
"""

from typing import Any, Iterable, List, Tuple, Type
import torch
from torch import nn


def freeze_bn(model: nn.Module):
    """
    Fix statistics and parameters of bn modules in the model
    """
    BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for m in model.modules():
        if m.training and isinstance(m, BN_MODULE_TYPES):
            m.requires_grad_(False)
            m.eval()
    bn_train = [m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)]
    assert len(bn_train) == 0