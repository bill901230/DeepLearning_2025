import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
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
        x = self.initial(x)  # (64, H/4, W/4)
        enc1 = self.layer1(x)  # (64, H/4, W/4)
        enc2 = self.layer2(enc1)  # (128, H/8, W/8)
        enc3 = self.layer3(enc2)  # (256, H/16, W/16)
        enc4 = self.layer4(enc3)  # (512, H/32, W/32)
        return x, enc1, enc2, enc3, enc4

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.encoder = ResNetEncoder()

        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dec5 = DecoderBlock(512+512, 512)
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        enc0, enc1, enc2, enc3, enc4 = self.encoder(x)

        bridge = self.bridge(enc4) # (1024, H/64, W/64)

        dec5 = self.dec5(bridge, enc4) # (512, H/32, W/32)
        dec4 = self.dec4(dec5, enc3) # (256, H/16, W/16)
        dec3 = self.dec3(dec4, enc2) # (128, H/8, W/8)
        dec2 = self.dec2(dec3, enc1) # (64, H/4, W/4)
        dec1 = self.dec1(dec2, enc0) # (32, H/2, W/2)


        return self.final_conv(dec1)
