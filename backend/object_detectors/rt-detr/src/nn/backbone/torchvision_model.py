"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import torch
import torch.nn as nn
import torchvision

from ...core import register
from .common import FrozenBatchNorm2d
from .utils import IntermediateLayerGetter

__all__ = ["TorchVisionModel"]


@register()
class TorchVisionModel(torch.nn.Module):
    def __init__(
        self,
        name,
        return_layers,
        weights=None,
        pretrained_weights=None,
        freeze=False,
        freeze_norm=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if weights is not None:
            weights = getattr(torchvision.models.get_model_weights(name), weights)

        model = torchvision.models.get_model(name, weights=weights, **kwargs)

        if pretrained_weights is not None:
            self._load_external_weights(model, pretrained_weights)

        # TODO hard code.
        if hasattr(model, "features"):
            model = IntermediateLayerGetter(model.features, return_layers)
        else:
            model = IntermediateLayerGetter(model, return_layers)

        self.model = model
        self._frozen = freeze

        if freeze or freeze_norm:
            self._freeze_norm(self.model)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _load_external_weights(model: torch.nn.Module, path: str) -> None:
        # Handles MoCo-v2 style checkpoints (e.g. geography-aware-ssl FMoW weights)
        # as well as plain torchvision state_dicts.
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        prefixes = ("module.encoder_q.", "encoder_q.", "module.backbone.", "backbone.", "module.")
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("module.encoder_k.") or k.startswith("encoder_k."):
                continue
            if "queue" in k:
                continue
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    break
            if nk.startswith("fc."):
                continue
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[TorchVisionModel] loaded pretrained weights from {path}")
        print(f"  {len(cleaned)} tensors mapped, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"  missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"  unexpected (first 10): {unexpected[:10]}")

    @staticmethod
    def _freeze_norm(module: nn.Module) -> nn.Module:
        if isinstance(module, nn.BatchNorm2d):
            return FrozenBatchNorm2d(module.num_features)
        for name, child in module.named_children():
            replaced = TorchVisionModel._freeze_norm(child)
            if replaced is not child:
                setattr(module, name, replaced)
        return module

    def train(self, mode: bool = True):
        # Keep a fully-frozen backbone in eval() regardless of parent.train()
        # so dropout / BN stats do not change during training.
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def forward(self, x):
        return self.model(x)


# TorchVisionModel('swin_t', return_layers=['5', '7'])
# TorchVisionModel('resnet34', return_layers=['layer2','layer3', 'layer4'])

"""
TorchVisionModel:
    name: swin_t
    return_layers: ['5', '7']
    weights: DEFAULT


model:
    type: TorchVisionModel
    name: resnet34
    return_layers: ['layer2','layer3', 'layer4']
    weights: DEFAULT
"""
