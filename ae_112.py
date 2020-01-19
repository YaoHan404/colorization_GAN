import torch as t
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.1
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None, relu=True):
        super(ResBlock, self).__init__()
        self.relu = relu
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel,momentum=momentum),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum)
        )
        self.shortcut = shortcut

    def forward(self, input):
        out = self.left(input)
        if self.shortcut is not None:
            out += self.shortcut(input)
        else:
            out += input
        if self.relu:
            out = F.leaky_relu(out, 0.01)
        else:
            out = F.tanh(out)
        return out


class ResBlock_T(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None, relu=True):
        super(ResBlock_T, self).__init__()
        self.relu = relu
        self.left = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum)
        )
        self.shortcut = shortcut

    def forward(self, input):
        out = self.left(input)
        if self.shortcut is not None:
            out += self.shortcut(input)
        else:
            out += input
        if self.relu:
            out = F.leaky_relu(out, 0.01)
        else:
            out = F.tanh(out)
        return out

class AE(nn.Module):
    def __init__(self, code_len, perturb=False):
        super(AE, self).__init__()
        self.perturb = perturb

        self.preprocess = nn.Sequential(
            nn.Dropout(0.1)
        )
        self.encoder_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=momentum),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        self.encoder_b = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=momentum),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        self.encoder_d = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80, momentum=momentum),
            nn.Tanh()
        )

        self.encoder_fc = nn.Linear(7*7*80, code_len)
        self.decoder_fc = nn.Linear(code_len, 7*7*80)

        self.decoder_d = self.stage_T(80, 64, 3, 2)
        self.decoder_c = self.stage_T(64, 32, 3, 2)
        self.decoder_b = self.stage_T(32, 16, 3, 2)
        self.decoder_t = self.stage_T(16, 16, 3, 2)
        self.decoder_a = self.stage(16, 3, 3, 1, False)   # 三通道输出

    def forward(self, input):
        corrupt_input = self.preprocess(input)      # 112
        e_a = self.encoder_a(corrupt_input)         # 56
        e_b = self.encoder_b(e_a)                   # 28
        e_c = self.encoder_c(e_b)                   # 14
        e_d = self.encoder_d(e_c)                   # 7

        e_d_flat = e_d.view(e_d.size()[0], -1)
        e_fc = self.encoder_fc(e_d_flat)
        #print 'before perturb ', t.mean(e_fc), t.var(e_fc)
        if self.perturb:
            e_fc += t.randn(e_fc.size()).cuda()
        #print 'after perturb ', t.mean(e_fc), t.var(e_fc)
        d_fc = self.decoder_fc(e_fc)
        d_fc_reshape = d_fc.view(e_d.size()[0], e_d.size()[1], e_d.size()[2], e_d.size()[3])    # like e_d

        d_d = self.decoder_d(d_fc_reshape)  # like e_c
        d_c = self.decoder_c(d_d)           # like e_b
        d_b = self.decoder_b(d_c)           # like e_a
        d_t = self.decoder_t(d_b)
        d_a = self.decoder_a(d_t)           # like input

        return [input, e_a, e_b, e_c, e_d], [d_a, d_b, d_c, d_d, d_fc_reshape]


    def stage(self, in_channel, out_channel, block_num, stride=1, relu=True):

        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum)
        )
        layers = []
        if block_num > 1:
            layers.append(ResBlock(in_channel, out_channel, stride, shortcut))
        else:
            layers.append(ResBlock(in_channel, out_channel, stride, shortcut, relu))
        for i in range(1, block_num):
            if i == block_num-1:
                layers.append(ResBlock(out_channel, out_channel, relu=relu))
            else:
                layers.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def stage_T(self, in_channel, out_channel, block_num, stride=1, relu=True):

        shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum)
        )
        layers = []
        layers.append(ResBlock_T(in_channel, out_channel, stride, shortcut))
        for i in range(1, block_num):
            if i == block_num-1:
                layers.append(ResBlock(out_channel, out_channel, relu=relu))
            else:
                layers.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*layers)



def plain_parameters_func(net, lr):
    return net.parameters()
