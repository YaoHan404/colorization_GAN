import torch as t
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.9
drop_out = '1d'
class MultiScale_Mix_sp3_s(nn.Module):
    def __init__(self):
        super(MultiScale_Mix_sp3_s, self).__init__()
        self.ori_scale_layers()
        self.half_scale_layers()
        self.demisemi_scale_layers()
        self.output_layers()

    def ori_scale_layers(self):
        self.ori_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU()
        )
        self.ori_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU()
        )
        self.ori_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU()
        )
        self.ori_exp_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU()
        )
        self.ori_exp_conv5 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )
        self.ori_exp_fc = nn.Sequential(
            nn.Linear(7*7*128, 128)
        )

    def half_scale_layers(self):
        self.half_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU()
        )
        self.half_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU()
        )
        self.half_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU()
        )
        self.half_exp_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU()
        )
        self.half_exp_conv5 = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )
        self.half_exp_fc = nn.Sequential(
            nn.Linear(3 * 3 * 64, 64)
        )
    def demisemi_scale_layers(self):
        self.demisemi_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU()
        )
        self.demisemi_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU()
        )
        self.demisemi_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU()
        )
        self.demisemi_exp_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU()
        )
        self.demisemi_exp_conv5 = nn.Sequential(
            nn.Conv2d(512, 32, 1),
            nn.BatchNorm2d(32, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )
        self.demisemi_exp_fc = nn.Sequential(
            nn.Linear(1 * 1 * 32, 32)
        )


    def output_layers(self):
        self.demisemi_exp_out = nn.Sequential(
            nn.Linear(32, 8)
        )
        self.half_exp_out = nn.Sequential(
            nn.Linear(64, 8)
        )
        self.ori_exp_out = nn.Sequential(
            nn.Linear(128, 8)
        )
        self.fusion_exp_out = nn.Sequential(
            nn.Linear(224, 8)
        )

    def ori_forward(self, input):
        out = self.ori_conv1(input)         # 224 64
        out = F.max_pool2d(out, (2, 2))     # 112 64

        out = self.ori_conv2(out)           # 112 128
        out = F.max_pool2d(out, (2, 2))     # 56 128

        out_u = self.ori_conv3(out)           # 56 256
        out = F.max_pool2d(out_u, (2, 2))     # 28 256

        out_exp_4 = self.ori_exp_conv4(out)           # 28 512
        out_exp = F.max_pool2d(out_exp_4, (2, 2))     # 14 512

        out_exp_5 = self.ori_exp_conv5(out_exp)           # 14 128
        out_exp = out_exp_5.view(out_exp_5.size()[0], -1)
        out_exp = self.ori_exp_fc(out_exp)      # 128

        return [out_u, out_exp_4, out_exp_5], out_exp

    def half_forward(self, input):
        out = F.upsample(input, size=(56, 56), mode='bilinear')

        out = self.half_conv1(out)        # 112 64
        out = F.max_pool2d(out, (2, 2))     # 56 64

        out = self.half_conv2(out)          # 56 128
        out = F.max_pool2d(out, (2, 2))     # 28 128

        out_u = self.half_conv3(out)          # 28 256
        out = F.max_pool2d(out_u, (2, 2))     # 14 256

        out_exp_4 = self.half_exp_conv4(out)  # 14 512
        out_exp = F.max_pool2d(out_exp_4, (2, 2))  # 7 512

        out_exp_5 = self.half_exp_conv5(out_exp)  # 7 128
        out_exp = out_exp_5.view(out_exp_5.size()[0], -1)
        out_exp = self.half_exp_fc(out_exp)  # 128

        return [out_u, out_exp_4, out_exp_5], out_exp

    def demisemi_forward(self, input):
        out = F.upsample(input, size=(28, 28), mode='bilinear')

        out = self.demisemi_conv1(out)          # 56 64
        out = F.max_pool2d(out, (2, 2))         # 28 64

        out = self.demisemi_conv2(out)          # 28 128
        out = F.max_pool2d(out, (2, 2))         # 14 128

        out_u = self.demisemi_conv3(out)          # 14 256
        out = F.max_pool2d(out_u, (2, 2))         # 7 256

        out_exp_4 = self.demisemi_exp_conv4(out)  # 7 512
        out_exp = F.max_pool2d(out_exp_4, (2, 2))  # 3 512

        out_exp_5 = self.demisemi_exp_conv5(out_exp)  # 3 128
        out_exp = out_exp_5.view(out_exp_5.size()[0], -1)
        out_exp = self.demisemi_exp_fc(out_exp)  # 128

        return [out_u, out_exp_4, out_exp_5], out_exp

    def concat_forward(self, ori, half, demisemi):
        out_exp = t.cat([ori, half, demisemi], 1)      # 224

        return out_exp

    def forward(self, input):
        ori_vec = self.ori_forward(input)

        half_vec = self.half_forward(input)

        demisemi_vec = self.demisemi_forward(input)

        fusion_vec = self.concat_forward(ori_vec[1], half_vec[1], demisemi_vec[1])
        fusion_exp_predict = self.fusion_exp_out(fusion_vec)
        fusion_exp_predict = F.softmax(fusion_exp_predict)

        return [ori_vec[0][0], ori_vec[0][1], ori_vec[0][2],
                half_vec[0][0], half_vec[0][1], half_vec[0][2],
                demisemi_vec[0][0], demisemi_vec[0][1], demisemi_vec[0][2],
                fusion_vec], [fusion_exp_predict]


def mix_parameters_func(net, lr):
    setting = [{'params': net.module.ori_conv1.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.ori_conv2.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.ori_conv3.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.ori_exp_conv4.parameters()},
               {'params': net.module.ori_exp_conv5.parameters()},
               {'params': net.module.half_conv1.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.half_conv2.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.half_conv3.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.half_exp_conv4.parameters()},
               {'params': net.module.half_exp_conv5.parameters()},
               {'params': net.module.demisemi_conv1.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.demisemi_conv2.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.demisemi_conv3.parameters(), 'lr': lr * 5e-1},
               {'params': net.module.demisemi_exp_conv4.parameters()},
               {'params': net.module.demisemi_exp_conv5.parameters()},
               {'params': net.module.ori_exp_fc.parameters()},
               {'params': net.module.ori_exp_out.parameters()},
               {'params': net.module.half_exp_fc.parameters()},
               {'params': net.module.half_exp_out.parameters()},
               {'params': net.module.demisemi_exp_fc.parameters()},
               {'params': net.module.demisemi_exp_out.parameters()},
               {'params': net.module.fusion_exp_out.parameters()},
               ]
    return setting

def plain_parameters_func(net, lr):
    return net.parameters()
