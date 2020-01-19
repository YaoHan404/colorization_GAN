from load_data import TrainData
from torch.utils.data import DataLoader
import time
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ae_112 import AE
from torch.autograd import Variable
from pure_disc_112 import ResNet_disc
from ems_perc import MultiScale_Mix_sp3_s
from torch import optim
import pure_disc_112
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tensorboardX import SummaryWriter
import tools
# 自动显示每行torch数据的信息
# 使用：在def 函数 前加一行@torchsnooper.snoop()
import torchsnooper
# from tensorboard_logger import Logger


class Trainer(object):

    def __init__(self):
        # 确定模型、目标函数、优化器
        self.generator_model = AE
        self.discriminator_model = ResNet_disc
        self.feature_map_model = MultiScale_Mix_sp3_s
        self.BATCH_SIZE = 64
        self.RUN_EPOCH = 100
        self.train_dataset = TrainData('indoorCVPR_09/Images/', train=True, use_processed_data=True)
        self.val_dataset = TrainData('indoorCVPR_09/Images/', train=False, use_processed_data=True)

    def train(self):
        discriminator = self.discriminator_model().cuda()
        feature_map = self.feature_map_model().cuda()
        generator = self.generator_model(64).cuda()

        generator_net = nn.DataParallel(generator, device_ids=[0, 1])  # use gpu0 and gpu1
        discriminator_net = nn.DataParallel(discriminator, device_ids=[0, 1])  # use gpu0 and gpu1
        feature_map_net = nn.DataParallel(feature_map, device_ids=[0, 1])  # use gpu0 and gpu1
        generator_net_optim = optim.RMSprop(generator_net.parameters(), lr=1e-5, alpha=0.9)
        discriminator_net_optim = optim.RMSprop(pure_disc_112.plain_parameters_func(discriminator_net, 1e-5), lr=1e-5, alpha=0.9)
        # self.actor.load_state_dict(torch.load('pre_model/7act_pretrain.pkl'))
        generator_net.cuda()

        feature_map_net.cuda()
        feature_map_net.eval()  # 特征提取网络不参与训练

        # 读取数据
        print('loading data...')
        train_loader = DataLoader(self.train_dataset,  # dataloader是可迭代对象
                                  batch_size=self.BATCH_SIZE,
                                  shuffle=True,  # 是否将数据打乱 , 可以用设置权重的方法控制一个batch中不同label数据的比例
                                  num_workers=0,  # 是否多线程加载数据，几个线程
                                  drop_last=True)  # 数据集中batch_size的非整数倍部分，是否丢弃
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.BATCH_SIZE,
                                shuffle=False,  # 是否将数据打乱 , 可以用设置权重的方法控制一个batch中不同label数据的比例
                                num_workers=0,  # 是否多线程加载数据，几个线程
                                drop_last=True)  # 数据集中batch_size的非整数倍部分，是否丢弃
        print('data loaded!')

        gen_stage = 0
        disc_stage = 0
        tensorboard_recoder_step = 0
        tensorboard_recoder_step_val = 0
        max_validation_acc = 0
        best_epoch = -1
        gen_lr = 1e-5
        disc_lr = 1e-5
        gen_optimizer = None
        disc_optimizer = None
        tb_writer = SummaryWriter('runs/AE_no_fc')
        for epoch in range(self.RUN_EPOCH):
            print('epoch:', epoch+1)
            epoch_start_time = time.time()

            generator_net.train()
            discriminator_net.train()
            tensorboard_recoder_step = self.train_an_epoch(generator_net, discriminator_net, feature_map_net,
                                generator_net_optim, discriminator_net_optim, train_loader,
                                tb_writer, epoch, tensorboard_recoder_step)

            generator_net.eval()
            discriminator_net.eval()
            tensorboard_recoder_step_val = self.validate_an_epoch(generator_net, discriminator_net, feature_map_net,
                                   val_loader, tb_writer,
                                   tensorboard_recoder_step_val, epoch)

            # self.save(epoch)

            epoch_time = time.time() - epoch_start_time
            print('time:{:.1f}'.format(epoch_time))

        print('===============training complete================')

    def train_an_epoch(self, gen_net, disc_net, feature_net,
                       gen_optim, disc_optim, train_loader,
                       tb_writer, epoch, tensorboard_recoder_step):
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_disc_adv_loss = 0
        total_gen_adv_loss = 0
        total_real_adv_acc = 0
        total_fake_adv_acc = 0
        total_gen_ae_loss = 0
        total_gen_perc_loss = 0
        for i, (img_gray, img_rbg) in enumerate(train_loader):
            img_gray = img_gray.cuda()
            img_gray = Variable(img_gray)
            img_rbg = img_rbg.cuda()
            img_rbg = Variable(img_rbg)

            # 限制特征提取网络的参数范围
            for p in disc_net.parameters():
                p.data.clamp_(-0.01, 0.01)

            gen_optim.zero_grad()
            disc_optim.zero_grad()
            feature_net.zero_grad()

            img_fake = gen_net(img_gray)

            if i % 100 == 0:
                tb_writer.add_image('real_image', (img_rbg.data[0]+1)/2, tensorboard_recoder_step)
                tb_writer.add_image('gray_image', (img_gray.data[0]+1)/2, tensorboard_recoder_step)
                tb_writer.add_image('fake_image', (img_fake[1][0].data[0]+1)/2, tensorboard_recoder_step)

            img_rgb_disc_output = disc_net(img_rbg)
            img_fake_disc_output = disc_net(img_fake[1][0].detach())

            # tensorboard写入图片
            # if i % 200 == 0:

            # discriminator loss
            # loss_disc_rgb + loss_disc_gray
            loss_disc_rgb = img_rgb_disc_output.mean(0)    # loss2
            loss_disc_gray = img_fake_disc_output.mean(0)*-1   # loss1

            # generator loss
            # loss_disc_gray_gen +
            img_fake_disc_output = disc_net(img_fake[1][0])  # 这里没有detach
            loss_disc_fake_gen = img_fake_disc_output.mean(0)*-1   # loss3

            img_rgb_feature_output = feature_net(img_rbg)
            img_fake_feature_output = feature_net(img_fake[1][0])

            #########################为什么用自己写的loss，这些loss是怎么来的，优点是什么

            loss_feature = 0.0
            perc_layer_weights = [500, 0, 0, 300, 0, 0, 300, 0, 0, 0]   # 怎么来的
            for fake_perc, real_perc, weight in \
                    zip(img_fake_feature_output[0], img_rgb_feature_output[0], perc_layer_weights):
                loss_feature += self.cal_feature_loss(fake_perc, real_perc.detach_())*weight

            # generator生成的fake图像和原本的rgb图像计算loss
            loss_gen_ae = self.cal_ae_loss(img_fake[1][0], img_rbg)

            '''
            训练discriminator
            每隔一段时间训练一次discriminator
            因为相对于generator，discriminator太容易训练
            '''
            if i % 10 == 0:
                loss_disc = loss_disc_gray + loss_disc_rgb
                loss_disc.backward()
                disc_optim.step()
            '''
            训练generator
            '''
            loss_gen = loss_disc_fake_gen*0 + loss_gen_ae*1 + loss_feature*0
            loss_gen.backward()
            gen_optim.step()

            real_adv_pred = tools.gan_predictor(img_rgb_disc_output)
            fake_adv_pred = tools.gan_predictor(img_fake_disc_output)
            # fake_cls_pred = tools.max_predict(img_fake_feature_output[1][0])
            # real_cls_pred = tools.max_predict(img_rgb_feature_output[1][0])

            real_adv_acc = tools.gan_record(real_adv_pred, real=True)
            fake_adv_acc = tools.gan_record(fake_adv_pred, real=False)
            # fake_cls_acc = tools.cls_record(fake_cls_pred, label)
            # real_cls_acc = tools.cls_record(real_cls_pred, label)
            # total_gen_adv_loss += loss_disc_fake_gen.item()  # loss3
            # total_disc_adv_loss += loss_disc_gray.item() + loss_disc_rgb.item()  # loss1 + loss2
            # total_real_adv_acc += real_adv_acc
            # total_fake_adv_acc += fake_adv_acc
            #
            # total_gen_ae_loss += loss_gen_ae.item()  # ？？？？？？？？？？？？？？？？？？？？？what
            # total_gen_perc_loss += loss_feature.item()

            tb_writer.add_scalar('acc/real_adv', real_adv_acc, tensorboard_recoder_step)
            tb_writer.add_scalar('acc/fake_adv', fake_adv_acc, tensorboard_recoder_step)
            tb_writer.add_scalar('loss/gen_adv', loss_disc_fake_gen.item(), tensorboard_recoder_step)
            tb_writer.add_scalar('loss/disc_adv', (loss_disc_gray+loss_disc_rgb).item(), tensorboard_recoder_step)
            tb_writer.add_scalar('loss/gen_ae', loss_gen_ae.item(), tensorboard_recoder_step)
            tb_writer.add_scalar('loss/feature', loss_feature.item(), tensorboard_recoder_step)

            tensorboard_recoder_step += 1
        return tensorboard_recoder_step


    def validate_an_epoch(self, generator_net, discriminator_net, feature_map_net,
                          val_loader, tb_writer,
                          tensorboard_recoder_step_val, epoch):
        for i, (img_gray, img_rbg) in enumerate(val_loader):
            img_gray = img_gray.cuda()
            img_gray = Variable(img_gray)
            img_rbg = img_rbg.cuda()
            img_rbg = Variable(img_rbg)

            img_fake = generator_net(img_gray)

            if i % 20 == 0:
                tb_writer.add_image('Val/real_image', (img_rbg.data[0]+1)/2, tensorboard_recoder_step_val)
                tb_writer.add_image('Val/gray_image', (img_gray.data[0]+1)/2, tensorboard_recoder_step_val)
                tb_writer.add_image('Val/fake_image', (img_fake[1][0].data[0]+1)/2, tensorboard_recoder_step_val)
                tensorboard_recoder_step_val += 1
            return tensorboard_recoder_step_val


    def cal_feature_loss(self, fake_feature, real_feature):
        size = fake_feature.size()
        num = 1
        for s in size:
            num *= s
        loss = torch.sum(torch.pow(fake_feature-real_feature, 2))
        loss /= num
        return loss


    def cal_ae_loss(self, fake, rgb, all=False):
        loss = 0
        if all:
            for o, l in zip(output, label):
                #loss += self.criterion(o, l)
                loss += torch.sum(torch.pow(o-l, 2))/(o.size()[0]*o.size()[1]*o.size()[2]*o.size()[3])
        else:
            # loss = 0.3 * t.sum(t.pow(output[0]-label[0], 2)) /\
            #        (output[0].size()[0]*output[0].size()[1]*output[0].size()[2]*output[0].size()[3])
            #  除以64个batch，3个通道，112*112的总像素数
            loss += torch.sum(torch.abs(fake-rgb)) /\
                    (fake.size()[0]*fake.size()[1]*fake.size()[2]*fake.size()[3])
            # print("torch.sum(torch.abs(fake-rgb))",torch.sum(torch.abs(fake-rgb)))
            # print("output[0].size()[0]",fake.size()[0])
            # print("output[0].size()[1]",fake.size()[1])
            # print("output[0].size()[2]",fake.size()[2])
            # print("output[0].size()[3]",fake.size()[3],loss)
        return loss


    def save(self, epoch):
        torch.save(self.actor.state_dict(), 'pre_model/' + str(epoch) + 'act_pretrain.pkl')  # save parameters of actor
        # #   actor.load_state_dict(torch.load('act_pretrain.pkl'))  # restore parameters




if __name__ == '__main__':
    # gen = AE
    # print(gen.parameters(64))

    pretrain = Trainer()
    # pretrain.actor.load_state_dict(torch.load('pre_model/20act_pretrain.pkl'))
    pretrain.train()

    # train_dataset = PreTrainData('/home/yunle/img/', train=True, use_processed_data=False)
    # train_dataset.__getitem__(0)