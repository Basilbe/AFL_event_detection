"""Pure PyTorch implementation of the R(2+1)D architecture."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class SpatioTemporalConv(nn.Module):
    """Factorised 3D convolution as described in the R(2+1)D paper."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        mid_planes: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if mid_planes is None:
            mid_planes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)

        spatial_stride = stride if isinstance(stride, tuple) else (1, stride, stride)
        temporal_stride = stride if isinstance(stride, tuple) else (stride, 1, 1)

        self.spatial_conv = nn.Conv3d(
            in_planes,
            mid_planes,
            kernel_size=(1, 3, 3),
            stride=spatial_stride,
            padding=(0, 1, 1),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm3d(mid_planes)
        self.temporal_conv = nn.Conv3d(
            mid_planes,
            out_planes,
            kernel_size=(3, 1, 1),
            stride=temporal_stride,
            padding=(1, 0, 0),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.relu(self.spatial_bn(self.spatial_conv(x)))
        x = self.relu(self.temporal_bn(self.temporal_conv(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = SpatioTemporalConv(in_planes, out_planes, stride=stride)
        self.conv2 = SpatioTemporalConv(out_planes, out_planes)

        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=(stride, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_planes),
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class R2Plus1DStem(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = SpatioTemporalConv(in_channels, 64, stride=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(x)
        x = self.pool(x)
        return x


class R2Plus1DBackbone(nn.Module):
    def __init__(self, layer_sizes: List[int], in_channels: int = 3) -> None:
        super().__init__()
        self.stem = R2Plus1DStem(in_channels)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layer_sizes[0])
        self.layer2 = self._make_layer(128, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(256, layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(512, layer_sizes[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(ResidualBlock(self.inplanes, planes, stride=stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class R2Plus1DClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        layer_sizes: Optional[List[int]] = None,
        dropout: float = 0.5,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [2, 2, 2, 2]
        self.backbone = R2Plus1DBackbone(layer_sizes, in_channels=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_r2plus1d_classifier(num_classes: int, **kwargs) -> R2Plus1DClassifier:
    """Factory for the default R(2+1)D-18 style classifier."""

    return R2Plus1DClassifier(num_classes=num_classes, **kwargs)
