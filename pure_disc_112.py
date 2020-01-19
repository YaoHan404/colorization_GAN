import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = shortcut

    def forward(self, input):
        out = self.left(input)
        if self.shortcut is not None:
            out += self.shortcut(input)
        else:
            out += input
        out = F.relu(out)
        return out

class ResNet_disc(nn.Module):
    def __init__(self):
        super(ResNet_disc, self).__init__()

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage_1 = self.stage(64, 64, 2, stride=1)
        self.stage_2 = self.stage(64, 128, 2, stride=2)
        self.stage_3_disc = self.stage(128, 128, 2, stride=2)
        self.stage_4_disc = self.stage(128, 128, 2, stride=2)

        self.dropout = nn.Sequential(
            nn.Dropout()
        )

        self.fc_disc = nn.Linear(128, 1, bias=False)

    def stage(self, in_channel, out_channel, block_num, stride=1):

        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = []
        layers.append(ResBlock(in_channel, out_channel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)                     # 112 64
        x = F.max_pool2d(x, (2, 2))         # 56 64

        x = self.stage_1(x)                  # 56 64
        x = self.stage_2(x)                  # 28 128

        x_disc = self.stage_3_disc(x)
        x_disc = self.stage_4_disc(x_disc)

        x_disc = self.dropout(x_disc)

        x_disc = F.avg_pool2d(x_disc, 4)
        x_disc = x_disc.view(x_disc.size(0), -1)
        x_disc = self.fc_disc(x_disc)
        return x_disc


def plain_parameters_func(net, lr):
    return net.parameters()

def mix_parameters_func(net, lr):
    setting = [{'params': net.module.preprocess.parameters()},
               {'params': net.module.stage_1.parameters()},
               {'params': net.module.stage_2.parameters()},
               {'params': net.module.stage_3_disc.parameters()},
               {'params': net.module.stage_4_disc.parameters()},
               {'params': net.module.fc_disc.parameters()}
               ]
    return setting