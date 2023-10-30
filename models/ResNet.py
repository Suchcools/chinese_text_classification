# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        datasetpath = '/home/dengrui/PyProjects/Multilabel/Chinese_Text_Classification_Pytorch/'
        self.model_name = 'TextCNN'
        self.train_path = datasetpath + dataset + '/data/train.txt'  # 训练集
        self.dev_path = datasetpath + dataset + '/data/dev.txt'  # 验证集
        self.test_path = datasetpath + dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            datasetpath + dataset + '/data/class_2.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = datasetpath + dataset + '/data/vocab.pkl'  # 词表
        self.save_path = datasetpath + dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = datasetpath + dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(datasetpath + dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.35  # 随机失活
        self.require_improvement = 200  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 1  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 50  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 2200  # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.002  # 学习率
        self.embed = 100  # self.embedding_pretrained.size(1)\
        # if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 5)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)

        # self.dropout = 0.2                                              # 随机失活
        # self.require_improvement = 200                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = 1                         # 类别数
        # self.n_vocab = 0                                                # 词表大小，在运行时赋值
        # self.num_epochs = 30                                            # epoch数
        # self.batch_size = 128                                           # mini-batch大小
        # self.pad_size = 1795                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 1e-3                                       # 学习率
        # self.embed = 100 #self.embedding_pretrained.size(1)\
        #     #if self.embedding_pretrained is not None else 300           # 字向量维度
        # self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        # self.num_filters = 256                                          # 卷积核数量(channels数)
        #


'''Convolutional Neural Networks for Sentence Classification'''


class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class Model(torch.nn.Module):
    def __init__(self,in_channels=1,classes=1):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool2d(3,2,1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    model=Model()
    model(torch.rand(64, 1, 2195, 100))
    print(21)