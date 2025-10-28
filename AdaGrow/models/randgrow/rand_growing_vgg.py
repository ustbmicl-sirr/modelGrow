import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class PlainBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=(3, 3)):
        super(PlainBlock, self).__init__()
        self.feature = nn.Conv2d(in_planes, planes, kernel_size, stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.norm(self.feature(x)))
        return out


class VGGNet4Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(VGGNet4Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.downstream1 = self._make_downstream(in_dim=64, out_dim=128)
        self.downstream2 = self._make_downstream(in_dim=128, out_dim=256)
        self.downstream3 = self._make_downstream(in_dim=256, out_dim=512)
        self.linear = nn.Linear(512, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.downstream1(out)
        out = self.layer2(out)
        out = self.downstream2(out)
        out = self.layer3(out)
        out = self.downstream3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGGNet3Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(VGGNet3Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.downstream1 = self._make_downstream(in_dim=64, out_dim=128)
        self.downstream2 = self._make_downstream(in_dim=128, out_dim=256)
        self.linear = nn.Linear(256, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))
        return nn.Sequential(*layers)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.downstream1(out)
        out = self.layer2(out)
        out = self.downstream2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_rand_growing_vgg(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return VGGNet3Block(PlainBlock, depths, num_classes=num_classes, image_channels=image_channels)
    return VGGNet4Block(PlainBlock, depths, num_classes=num_classes, image_channels=image_channels)
