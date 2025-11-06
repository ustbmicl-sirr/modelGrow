import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.expand_size =  max(in_size // reduction, 8)
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MobileNetBlock(nn.Module):
    def __init__(self, kernel_size, size):
        super(MobileNetBlock, self).__init__()
        expand_size = 4 * size
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(size, expand_size, kernel_size=1, bias=False)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(expand_size, expand_size, (kernel_size, kernel_size), padding=padding, groups=expand_size)
        self.act2 = nn.SiLU(inplace=True)
        self.se = SeModule(expand_size)
        self.conv3 = nn.Conv2d(expand_size, size, kernel_size=1, bias=False)
        self.act3 = nn.SiLU(inplace=True)

    def forward(self, x):
        skip = x
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.se(out)
        out = self.conv3(out)
        return self.act3(out + skip)
  

class RuntimeMobileNet4Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(RuntimeMobileNet4Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1 = self._make_layer(block, num_blocks[0], kernel_size=3, size=16)
        self.layer2 = self._make_layer(block, num_blocks[1], kernel_size=5, size=32)
        self.layer3 = self._make_layer(block, num_blocks[2], kernel_size=5, size=64)
        self.layer4 = self._make_layer(block, num_blocks[3], kernel_size=5, size=96)
        self.downstream1 = self._make_downstream(in_dim=16, out_dim=32)
        self.downstream2 = self._make_downstream(in_dim=32, out_dim=64)
        self.downstream3 = self._make_downstream(in_dim=64, out_dim=96)
        self.linear = nn.Linear(96, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, kernel_size, size):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(kernel_size, size))
        return nn.Sequential(*layers)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
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


class RuntimeMobileNet3Block(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3):
        super(RuntimeMobileNet3Block, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1 = self._make_layer(block, num_blocks[0], kernel_size=3, size=32)
        self.layer2 = self._make_layer(block, num_blocks[1], kernel_size=5, size=64)
        self.layer3 = self._make_layer(block, num_blocks[2], kernel_size=5, size=96)
        self.downstream1 = self._make_downstream(in_dim=32, out_dim=64)
        self.downstream2 = self._make_downstream(in_dim=64, out_dim=96)
        self.linear = nn.Linear(96, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, kernel_size, size):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(kernel_size, size))
        return nn.Sequential(*layers)
    
    def _make_downstream(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.downstream1(out)
        out = self.layer2(out)
        out = self.downstream2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_runtime_mobilenetv3(depths, num_classes=10, image_channels=3):
    if len(depths) == 3:
        return RuntimeMobileNet3Block(MobileNetBlock, depths, num_classes=num_classes, image_channels=image_channels)
    return RuntimeMobileNet4Block(MobileNetBlock, depths, num_classes=num_classes, image_channels=image_channels)


if __name__ == "__main__":
    import torch
    model = get_runtime_mobilenetv3([1,2,3,4])
    x = torch.rand(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
