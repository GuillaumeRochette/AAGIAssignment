from typing import Tuple
from types import MethodType

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
import torch.utils.checkpoint as checkpoint

from timm.models import swin_transformer_v2
from torchvision.ops import FeaturePyramidNetwork


def basic_layer_forward(self, x: Tensor) -> Tensor:
    """
    Hacked method of the forward() of the BasicLayer within the SwinTransformerV2, where the downsample() has been moved out of the forward().

    :param self:
    :param x:
    :return:
    """
    for blk in self.blocks:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint.checkpoint(blk, x)
        else:
            x = blk(x)
        # x = self.downsample(x)
    return x


def swin_transformer_v2_forward_features(self, x: Tensor) -> Tuple[Tensor, ...]:
    """
    Hacked method of the forward_features() where all the feature maps are being returned instead of just the final one.
    This is also where the downsample() instead of within the layer.

    :param self:
    :param x:
    :return:
    """
    x = self.patch_embed(x)

    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    y = []
    for layer in self.layers:
        x = layer(x)
        # Hack start.
        n, l, c = x.shape
        h, w = layer.input_resolution
        z = x.permute(0, 2, 1).reshape(n, c, h, w)
        y.append(z)
        x = layer.downsample(x)
        # Hack end.

    # x = self.norm(x)  # B L C
    # return x
    return tuple(y)


class SwinTransformerV2Backbone(Module):
    """
    SwinTransformerV2Backbone where the multiscale feature maps are extracted from the input image.

    """
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
    ):
        super().__init__()
        # Select a backbone from a list of supported backbones.
        if backbone == "swinv2_tiny_window8_256":
            backbone = swin_transformer_v2.swinv2_tiny_window8_256(pretrained=pretrained)
        elif backbone == "swinv2_base_window8_256":
            backbone = swin_transformer_v2.swinv2_base_window8_256(pretrained=pretrained)
        # elif backbone == "xyz":
        #     backbone = ...
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        # "Hack" the backbone layers to not downsample the feature maps at the end of the block but rather after it.
        for layer in backbone.layers:
            layer.forward = MethodType(basic_layer_forward, layer)
        # "Hack" the backbone forward_features to return intermediate features.
        backbone.forward_features = MethodType(
            swin_transformer_v2_forward_features, backbone
        )
        # Transfer the modules and the forward_features from the "hacked" backbone to the current module.
        self._modules = backbone._modules
        self.forward_features = backbone.forward_features

        self.in_size = backbone.patch_embed.img_size
        self.out_size = [layer.input_resolution for layer in backbone.layers]

        self.in_channels = 3
        self.out_channels = [layer.dim for layer in self.layers]

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.forward_features(x)
        return x


class FeaturePyramidNetworkHead(FeaturePyramidNetwork):
    """
    Convenient wrapper for the FeaturePyramidNetwork, which does not requires a dict as input, but rather just a tuple.
    """
    def forward(self, x: Tuple[Tensor, ...]) -> Tensor:
        x = {i: z for i, z in enumerate(x)}
        x = super().forward(x)
        x = tuple(x.values())
        return x[0]


class FPNSwinTransformerV2(Module):
    """
    FPNSwinTransformerV2 which first extract the feature maps from the image using SwinTransformerV2Backbone,
    then combines the maps with FeaturePyramidNetworkHead to produce a high-resolution semantically rich feature map,
    before turning the final map into a semantic map.
    """
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        n_classes: int,
    ):
        super().__init__()

        self.backbone = SwinTransformerV2Backbone(
            backbone=backbone,
            pretrained=pretrained,
        )
        self.head = FeaturePyramidNetworkHead(
            in_channels_list=self.backbone.out_channels,
            out_channels=self.backbone.out_channels[0],
            norm_layer=BatchNorm2d,
        )
        self.out_layer = Conv2d(
            in_channels=self.backbone.out_channels[0],
            out_channels=n_classes,
            kernel_size=1,
        )

        self.in_size = self.backbone.in_size
        self.out_size = self.backbone.out_size[0]

        self.in_channels = self.backbone.in_channels
        self.out_channels = n_classes

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        x = self.out_layer(x)
        return x


def main():
    m = FPNSwinTransformerV2(
        backbone="swinv2_tiny_window8_256",
        pretrained=True,
        n_classes=19,
    )
    print(m.in_size)
    print(m.out_size)
    print(m.in_channels)
    print(m.out_channels)
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(y.shape)


if __name__ == "__main__":
    main()
