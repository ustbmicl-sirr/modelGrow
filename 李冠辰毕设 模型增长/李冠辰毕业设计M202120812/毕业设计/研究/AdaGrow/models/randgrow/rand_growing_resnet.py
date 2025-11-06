import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.feature1 = nn.Conv2d(planes, planes, (3, 3), stride=stride, padding=1)
        self.feature2 = nn.Conv2d(planes, planes, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.feature1(x)))
        out = F.relu(self.bn2(self.feature2(out)) + x)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.feature1 = nn.Conv2d(planes, planes//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.feature2 = nn.Conv2d(planes//4, planes//4, (3, 3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes//4)
        self.feature3 = nn.Conv2d(planes//4, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.feature1(x)))
        out = F.relu(self.bn2(self.feature2(out)))
        out = F.relu(self.bn3(self.feature3(out)) + x)
        return out


class ResNet4Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(ResNet4Block, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = 64
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
            layers.append(block(planes, stride))
        return nn.Sequential(*layers)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

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


class ResNet3Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(ResNet3Block, self).__init__()
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

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, stride))
        return nn.Sequential(*layers)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

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


def get_rand_growing_basic_resnet(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return ResNet3Block(BasicBlock, depths, num_classes=num_classes, image_channels=image_channels)
    return ResNet4Block(BasicBlock, depths, num_classes=num_classes, image_channels=image_channels)


def get_rand_growing_bottleneck_resnet(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return ResNet3Block(Bottleneck, depths, num_classes=num_classes, image_channels=image_channels)
    return ResNet4Block(Bottleneck, depths, num_classes=num_classes, image_channels=image_channels)
