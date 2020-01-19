import torch as t
from torch.autograd import Variable
import numpy as np

def gan_predictor(output):
    # [1,2,0,-0.5,-3] -> [0, 0, 0, 1, 1]
    # 找到小于0的部分
    return (output < 0).int().squeeze()

def max_predict(output):
    return t.argmax(output, dim=1)

'''
计算一个batch里面discriminator的准确率
'''
def gan_record(prediction, label=None, real=True):
    assert isinstance(prediction, t.Tensor)
    label = np.ones(prediction.size()[0], np.int) if real else np.zeros(prediction.size()[0], np.int)
    label = t.Tensor(label).long().cuda()
    correct_n = t.eq(prediction.int(), label.int()).sum().int()
    # self.correct_num += correct_n
    # self.sample_num += prediction.size()[0]

    return correct_n.float()/prediction.size()[0]

def cls_record(prediction, label):
    assert isinstance(prediction, t.Tensor)
    assert isinstance(label, t.Tensor)
    correct_n = t.eq(prediction.int(), label.int()).sum().int()
    # self.correct_num += correct_n
    # self.sample_num += prediction.size()[0]

    return correct_n.float()/prediction.size()[0]

# data = np.array([1,2,0,-0.5,-3])
# data_tensor = t.tensor(data)
# data_tensor = data_tensor.cuda()
# print(gan_record(gan_predictor(data_tensor),real=True))
# print(max_predict(data_tensor))