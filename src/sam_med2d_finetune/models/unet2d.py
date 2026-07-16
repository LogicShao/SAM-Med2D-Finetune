"""Conventional five-level 2D U-Net for the P0 baseline comparison.

Architecture follows the frozen definition in the baselines PRD:
- 4 input MRI channels
- Encoder: [32, 64, 128, 256, 512] with two 3x3 Conv-IN-LeakyReLU blocks
- 2x2 max pooling downsampling
- Decoder: 2x2 transposed conv upsampling + skip concatenation
- Output: 3 independent sigmoid heads (ET, TC, WT)
- No attention, residual blocks, pretrained encoders, or deep supervision
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 Conv -> InstanceNorm -> LeakyReLU block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """Standard 5-level 2D U-Net with independent sigmoid output heads.

    Frozen architecture per baselines PRD:
    - Encoder widths: 32, 64, 128, 256, 512
    - Two 3x3 Conv-IN-LeakyReLU per block
    - 2x2 max pooling down, 2x2 transposed conv up
    - Skip connections: encoder block output -> decoder block input
    - 3 independent ET/TC/WT sigmoid logit outputs
    """

    def __init__(self, in_channels=4, num_classes=3, base_width=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        widths = [base_width * (2 ** i) for i in range(5)]  # [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.enc5 = ConvBlock(widths[3], widths[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(widths[4], widths[3])  # skip concat: widths[3]+widths[3]

        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(widths[3], widths[2])

        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(widths[2], widths[1])

        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(widths[1], widths[0])

        # Output: 3 independent logit heads (ET, TC, WT)
        self.out_conv = nn.Conv2d(widths[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.out_conv(d1)  # (B, 3, H, W)
        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
