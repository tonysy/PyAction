from typing import Any, Iterable, List, Tuple, Type
import torch
from torch import nn
from pyaction.models.nonlocal_helper import Nonlocal

BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

# No need recursive, since .modules() returns all the modules
def freeze(model, freeze_bn_stats=True, freeze_nln=False):
    
    # First, freeze the whole model:
    # -grad
    for p in model.parameters():
        p.requires_grad = False
    # -bn stats
    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, BN_MODULE_TYPES):
                m.track_running_stats=False

    # then unfreeze non-local modules
    if not freeze_nln:
        for m in model.modules():
            if isinstance(m, Nonlocal):
                print(m, "!!!!!!!!!!!!!!!!!!!!!!")
                # -grad
                for p in m.parameters():
                    p.requires_grad = True
                # -bn stats
                for m_ in m.modules():
                    if isinstance(m_, BN_MODULE_TYPES):
                        m_.track_running_stats=True