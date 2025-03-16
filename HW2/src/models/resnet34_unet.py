import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 64
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet34: [3, 4, 6, 3] 個 block
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [ConvBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)  # (64, H/2, W/2)
        enc1 = self.layer1(x)  # (64, H/4, W/4)
        enc2 = self.layer2(enc1)  # (128, H/8, W/8)
        enc3 = self.layer3(enc2)  # (256, H/16, W/16)
        enc4 = self.layer4(enc2)  # (512, H/16, W/16)
        return enc1, enc2, enc3, enc4

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.encoder = ResNetEncoder()

        # self.bottleneck = ConvBlock(512, 512)

        self.dec4 = DecoderBlock(512+256, 256)
        self.dec3 = DecoderBlock(256+128, 128)
        self.dec2 = DecoderBlock(128+64, 64)
        self.dec1 = DecoderBlock(64+64, 32)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1, enc2, enc3, enc4 = self.encoder(x)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.final_conv(dec1)
