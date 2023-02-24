from collections import OrderedDict
import os

from .TT_FHE_SMALL import Block_TT
from .model_utils.netbin import SeqBinModelHelper, BinLinearPos, Binarize01Act, g_weight_binarizer, \
    activation_quantize_fn2, \
    BinConv2d, g_weight_binarizer3, setattr_inplace, BatchNormStatsCallbak, g_use_scalar_scale_last_layer, \
    InputQuantizer
from .model_utils.utils import ModelHelper, Flatten
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F






class Block_resnet_multihead_general_BN_vf_imgnet_v2(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=1, last=False):
        super(Block_resnet_multihead_general_BN_vf_imgnet_v2, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,30,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes, k = (6,5), stride=stride, padding=3,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes, k=(5,6), stride=stride,
                                                       padding=3,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes, k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        groupvf = 30
        T = 0.0
        print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)

        if last:
           self.Block_convf = Block_TT(self.cpt * in_planes, self.cpt*in_planes, k=1, stride=1,
                                          padding=0,
                                          groupsici=int(self.cpt * in_planes / groupvf), T=T, last=last)  # int(4*in_planes/4))
        else:
           self.Block_convf = Block_TT(self.cpt * in_planes, 2*out_planes, k=1, stride=1,
                                          padding=0,
                                          groupsici=int(self.cpt * in_planes / groupvf), T=T, last=last)  # int(4*in_planes/4))

        self.last = last

    def forward(self, x):
        """out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)"""
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            #pass
            out4 = self.Block_conv1.act(self.Block_conv4(x)-0.5)
            out3 = self.Block_conv1.act(self.Block_conv4(out3) - 0.5)
        else:
            out4 = x
        pad2 = nn.ZeroPad2d((0, 1, 0, 1))
        pad3 = nn.ZeroPad2d((0, 2, 0, 2))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 56) and (out1.shape[-1]==58):
            out4 = pad3(out4)
            out3 = pad3(out3)
            out1 = pad21(out1)
            out2 = pad31(out2)
        elif (x.shape[-1] == 56):
            out4 = self.pad0(out4)
            out3 = self.pad0(out3)
        elif (x.shape[-1] == 29):
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad3(out3)
            out4 = pad3(out4)
        elif (x.shape[-1] == 16):
            out3 = pad2(out3)
            out4 = pad2(out4)
        elif (x.shape[-1] == 9) and out1.shape[-1] == 6:
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad3(out3)
            out4 = pad3(out4)
        elif (x.shape[-1] == 58): #and out1.shape[-1] == 6:
            out3 = pad2(out3)
            out4 = pad2(out4)
        elif (x.shape[-1] == 30): #and out1.shape[-1] == 6:
            out3 = pad2(out3)
            out4 = pad2(out4)
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)


class TT_vf_19lv3_imgnet(SeqBinModelHelper, nn.Module, ModelHelper):
    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.make_small_network(self)

    #def _setup_network(self):
        #self.make_small_network(self)

    @classmethod
    def make_small_network(
            cls, self):
        n = self.args.nfilter
        t = self.args.tfilter
        p = n * t
        self.layers = [nn.AvgPool2d(2),
            nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        print(self.args.layers)
        # SUR 150 epochs
        if self.args.layers == 0:
            cfg = [(p,2) , (2 * p, 2)] #- Acc1: 44.902% Acc5: 68.556%
        elif self.args.layers == 1:
            cfg = [(p,2) , (2 * p, 2), (4 * p, 2)]
        elif self.args.layers == 2:
            cfg = [(p,2) , (2 * p, 2), (4 * p, 2), (8*p,2)]
        elif self.args.layers == 3:
            cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]
        #elif self.args.layers == 4:
        #    cfg = [p, (2 * p, 2), 2 * p, (4 * p, 2), (8 * p, 2)]

        self.layers.append(nn.BatchNorm2d(p))
        self.layers.append(Binarize01Act())

        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general_BN_vf_imgnet_v2(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = 1, stride=stride, last=last))
            in_planes = 2*out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(Classifier_scale(fcsize, 10, 1000))
        self.features = nn.Sequential(*self.layers)

    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 224, 224)).shape[1]  # image size: 64x32
        return int(size)

class Polynome_ACT(nn.Module):
    def __init__(self, alpha=0.47, beta=0.50, gamma=0.09):
        super().__init__()

    def forward(self, input):
        out = 0.47 + 0.50 * input + 0.09 * input ** 2  # - 1.7e-10 * input**3 #self.alpha * self.h_function((input / self.gamma) + self.beta) - self.alpha * self.h2_function(self.beta)
        return out

class Classifier_scale(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, fcsize, out_planes, inter=1000):
        super(Classifier_scale, self).__init__()

        self.lin1 = nn.Linear(fcsize, inter, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(inter)
        self.lin2 = nn.Linear(inter, 1000, bias=True)
        self.Polynome_ACT = Polynome_ACT(0, 0, 0)

    def forward(self, x):
        x = self.BN2(self.lin1(x))
        # if self.training:
        # x = F.relu(x)
        x = self.Polynome_ACT(x)
        # else:
        #    x = self.Polynome_ACT(x)
        return self.lin2(x)