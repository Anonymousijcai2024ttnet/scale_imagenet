from collections import OrderedDict
import os

from .TT_FHE import get_exp_with_y
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
import numpy as np
import pandas as pd
from sympy import symbols, SOPform, POSform, simplify_logic

class Block_resnet(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, Abit_inter = 2, k=3, t=8, padding=1, stride=1, groupsici=1, last=False):
        super(Block_resnet, self).__init__()
        #print(in_planes, groupsici)
        self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=padding, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.last = last

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        if self.last:
            out = F.gelu(self.bn2(self.conv2(out)))
        else:
            out = self.bn2(self.act(self.conv2(out)))
        return out



class Block_resnet_BN(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, Abit_inter = 2, k=3, t=8, padding=1, stride=1, groupsici=1, last=False):
        super(Block_resnet_BN, self).__init__()
        #print(in_planes, groupsici)
        self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=padding, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.last = last

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        if self.last:
            out = F.gelu(self.bn2(self.conv2(out)))
        else:
            out = self.act(self.bn2(self.conv2(out)))
        return out


class Block_resnet_big(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, Abit_inter = 2, k=3, t=8, padding=1, stride=1, groupsici=1, last=False):
        super(Block_resnet_big, self).__init__()
        #print(in_planes, groupsici)
        self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=padding, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv1b = nn.Conv2d(t * in_planes, t * in_planes, kernel_size=1,
                               stride=1, padding=0, groups=groupsici, bias=False)
        self.bn1b = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.last = last

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn1b(self.conv1b(out)))
        if self.last:
            out = F.gelu(self.bn2(self.conv2(out)))
        else:
            out = self.bn2(self.act(self.conv2(out)))
        return out

class Block_resnet_multihead_general(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                   self.Block_conv1 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 4, stride=stride, padding=1,
                                  groupsici=int(in_planes / g))  # int(in_planes/1))
                   self.cpt += 1
                elif index_g == 1:
                    self.Block_conv2 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 3, stride = stride, padding=1,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g+2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 2, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    self.Block_conv4 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 1, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
        if Abit_inter>1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        self.Block_convf = Block_resnet(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1, padding=1, last=True,
                                  groupsici=int(self.cpt * in_planes / g2))  # int(4*in_planes/4))
        self.stride = stride
        self.last = last

    def forward(self, x):
        out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)
        if (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
            out2 = out2[:,:,:-1,:-1]

        if self.cpt==4:
            outf = torch.cat((out1, out2, out3, out4), axis=1)
        elif self.cpt==2 and self.groups[1] is None:
            outf = torch.cat((out1, out3), axis=1)
        else:
            outf = torch.cat((out2, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)



class Block_resnet_multihead_general_big(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_big, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                   self.Block_conv1 = Block_resnet_big(in_planes, in_planes, Abit_inter=Abit_inter, k = 4, stride=stride, padding=1,
                                  groupsici=int(in_planes / g))  # int(in_planes/1))
                   self.cpt += 1
                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_big(in_planes, in_planes, Abit_inter=Abit_inter, k = 3, stride = stride, padding=1,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g+2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_big(in_planes, in_planes, Abit_inter=Abit_inter, k = 2, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    self.Block_conv4 = Block_resnet_big(in_planes, in_planes, Abit_inter=Abit_inter, k = 1, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
        if Abit_inter>1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        self.Block_convf = Block_resnet_big(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1, padding=1, last=True,
                                  groupsici=int(self.cpt * in_planes / g2))  # int(4*in_planes/4))
        self.stride = stride
        self.last = last

    def forward(self, x):
        out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)
        if (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
            out2 = out2[:,:,:-1,:-1]

        if self.cpt==4:
            outf = torch.cat((out1, out2, out3, out4), axis=1)
        elif self.cpt==2 and self.groups[1] is None:
            outf = torch.cat((out1, out3), axis=1)
        else:
            outf = torch.cat((out2, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)



class Block_resnet_multihead_general_8(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_8, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                   self.Block_conv1 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 8, stride=stride, padding=3,
                                  groupsici=int(in_planes / g))  # int(in_planes/1))
                   self.cpt += 1
                elif index_g == 1:
                    self.Block_conv2 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 4, stride = stride, padding=1,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g+2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 2, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    self.Block_conv4 = Block_resnet(in_planes, in_planes, Abit_inter=Abit_inter, k = 1, stride = stride, padding=0,
                                  groupsici = int(in_planes/g))#int(in_planes/2))
                    self.cpt += 1
        if Abit_inter>1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        self.Block_convf = Block_resnet(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=3, stride=1, padding=1, last=True,
                                  groupsici=int(self.cpt * in_planes / 6))  # int(4*in_planes/4))
        self.stride = stride
        self.last = last

    def forward(self, x):
        #print(x.shape)
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
            out2 = out2[:,:,:-1,:-1]
        #print(out1.shape, out2.shape, out3.shape)
        outf = torch.cat((out1, out2, out3), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class Block_resnet_multihead_general_BN(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    # self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 7, stride=stride, padding=4,
                    #               groupsici=int(in_planes / g))  # int(in_planes/1))
                    self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                                                       padding=0,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        # print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        if self.stride == 2:
            groupvf = 9
        else:
            groupvf = 12
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
        if self.stride == 2:
            out1 = self.Block_conv1(x)
        else:
            out1 = x
        # print(out1.shape, out2.shape, out3.shape, x.shape)
        # if (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
        #    out2 = out2[:,:,:-1,:-1]
        if (self.stride == 1 and x.shape[-1] == 16) or (self.stride == 2 and x.shape[-1] == 17) or (
                self.stride == 2 and x.shape[-1] == 9) \
                or (self.stride == 2 and x.shape[-1] == 5):
            out2 = out2[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]

        elif (self.stride == 2 and x.shape[-1] == 11) or (self.stride == 2 and x.shape[-1] == 7):
            out1 = out1[:, :, :-1, :-1]
        # print(out1.shape, out2.shape, out3.shape)
        # print(out1.shape, out2.shape, out3.shape, x.shape)
        outf = torch.cat((out1, out2, out3), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)



class TT_general(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]
        if self.args.layers ==18:
            cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]
        elif self.args.layers ==30:
            cfg = [p, (2 * p, 2), (4 * p, 2), 4*p, (8 * p, 2), 8*p]
        else:
            raise "pb"
        if self.args.Abit_inter > 1:
            self.layers.append(activation_quantize_fn2(self.args.Abit_inter))
        elif self.args.Abit_inter == 1:
            self.layers.append(Binarize01Act())
        else:
            raise "pb"
        self.layers.append(nn.BatchNorm2d(p))

        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)

class TT_general_big(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]
        if self.args.layers ==18:
            cfg = [(2 * p, 2), (4 * p, 2), (8 * p, 2)]
        elif self.args.layers ==30:
            cfg = [(2 * p, 2), (4 * p, 2), 4*p, (8 * p, 2), 8*p]
        else:
            raise "pb"
        if self.args.Abit_inter > 1:
            self.layers.append(activation_quantize_fn2(self.args.Abit_inter))
        elif self.args.Abit_inter == 1:
            self.layers.append(Binarize01Act())
        else:
            raise "pb"
        self.layers.append(nn.BatchNorm2d(p))

        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general_big(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)

class TT_general_8(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=1, padding=3, groups=1, bias=False)]
        if self.args.layers ==18:
            cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]
        elif self.args.layers ==30:
            cfg = [p, (2 * p, 2), (4 * p, 2), 4*p, (8 * p, 2), 8*p]
        else:
            raise "pb"
        if self.args.Abit_inter > 1:
            self.layers.append(activation_quantize_fn2(self.args.Abit_inter))
        elif self.args.Abit_inter == 1:
            self.layers.append(Binarize01Act())
        else:
            raise "pb"
        self.layers.append(nn.BatchNorm2d(p))

        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general_8(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_general_correctBN(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]
        if self.args.layers ==18:
            cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]
        elif self.args.layers ==30:
            cfg = [p, (2 * p, 2), (4 * p, 2), 4*p, (8 * p, 2), 8*p]
        else:
            raise "pb"
        self.layers.append(nn.BatchNorm2d(p))
        if self.args.Abit_inter > 1:
            self.layers.append(activation_quantize_fn2(self.args.Abit_inter))
        elif self.args.Abit_inter == 1:
            self.layers.append(Binarize01Act())
        else:
            raise "pb"


        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general_BN(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 512))
        self.layers.append(nn.Linear(512, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)

class Block_resnet_multihead_general_BN_vf_small(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_small, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,2,4,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 4, stride=stride, padding=2,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=3, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf = 4
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (self.stride==1 and x.shape[-1]==7) or (x.shape[-1] == 10) or (self.stride==1 and x.shape[-1]==8) or (self.stride==1 and x.shape[-1]==14) or (self.stride==1 and x.shape[-1]==9) or (self.stride==1 and x.shape[-1]==11)  or (self.stride==1 and x.shape[-1]==12) or (self.stride==1 and x.shape[-1]==20) or (self.stride==1 and x.shape[-1]==18) or (self.stride==1 and x.shape[-1]==16): #or (self.stride == 2 and x.shape[-1] == 13): #or (self.stride == 2 and x.shape[-1] == 13):
            #out1 = self.pad0(out1)
            #out4 = self.pad0(out4)
            out1 = out1[:,:,:-1,:-1]
            out3 = out3[:,:,:-1,:-1]
            #if (x.shape[-1] == 10):
            #    out4 = out4[:, :, :-1, :-1]
        elif (x.shape[-1] == 8) or (x.shape[-1] == 14)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            out1 = out1[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]
            out4 = out4[:, :, :-1, :-1]
            #out1 = self.pad0(out1)
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)

        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)


class Block_TT(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, k=3, t=8, padding=1, stride=1,
                 groupsici=1, quant_flag="float", blockici=0, T=0.0, last= False):
        super(Block_TT, self).__init__()
        self.k = k
        self.last = last
        self.blockici = blockici
        self.final_mask_noise = None
        self.in_planes = in_planes
        self.groupsici = groupsici
        self.pad1 = None
        if padding != 0:
            self.pad1 = nn.ConstantPad2d(padding, 0)
        if quant_flag == "bin":
            wb = g_weight_binarizer
            self.conv1 = BinConv2d(wb, in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=0, groups=groupsici, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                                  stride=stride, padding=0, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        self.act = Binarize01Act()

    def forward(self, x, compute_final_mask_noise = True):
        if self.final_mask_noise is not None and compute_final_mask_noise:
           x = self.final_mask_noise*x
        self.input_layer = x.clone()
        if self.pad1 is not None:
            x = self.pad1(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        if self.last:
            out = F.relu(self.bn2(self.conv2(out))) #(100000*F.relu(self.bn2(self.conv2(out)))).double().float()
            #self.last_F = out.clone()
        else:
            out = self.act(self.bn2(self.conv2(out)))
        self.output_layer = out.clone()
        return out

    def get_TT_block_all_filter(self, device, blockici, sousblockici):
        self.blockici = blockici
        self.sousblockici = sousblockici
        with torch.no_grad():
            nbrefilter = self.in_planes
            chanel_interest = int(nbrefilter/self.groupsici)
            self.n = self.k ** 2 * chanel_interest
            c_a_ajouter = nbrefilter - chanel_interest
            l = [[int(y) for y in format(x, 'b').zfill(self.n)] for x in range(2 ** self.n)]
            df = pd.DataFrame(l)
            self.df = df.reset_index()
            x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, chanel_interest, self.k,
                                             self.k)
            y = x_input_f2.detach().clone()
            padding = torch.autograd.Variable(y)
            for itera in range(c_a_ajouter):
                x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
            del padding
            if self.pad1 is not None:
                x_input_f2 = self.pad1(x_input_f2)
            self.res_numpy = self.forward(x_input_f2.to(device), compute_final_mask_noise=False).squeeze(-1).squeeze(-1).detach().cpu().clone().numpy()
        return self.res_numpy

    def get_TT_block_1filter(self, filterici, path_save_exp):
        self.filterici = filterici
        self.path_save_exp = path_save_exp
        resici = self.res_numpy[:, filterici]
        unique = np.unique(resici)
        print(unique)
        if len(unique) == 1:
            # s'il n'y a qune seule valeur, enregistre la valeur
            self.save_cnf_dnf(resici[0], str(resici[0]))
            table = np.chararray((2 ** self.n, 2 ** self.n), itemsize=3)
            table[:][:] = str(resici[0])
            np.save(self.path_save_exp + 'table_outputblock_' +
                    str(self.blockici) + '_filter_' + str(self.filterici) +
                    '_value_' + str(resici[0]) + '_coefdefault_' +
                    str(resici[0]) + '.npy', table)
            exp_CNF, exp_DNF, exp_CNF3 = None, None, None
        else:
            # sinon on cherche la formule SAT
            exp_CNF, exp_DNF, exp_CNF3 = self.iterate_over_filter(resici, unique)
        return exp_CNF, exp_DNF, exp_CNF3

    def save_cnf_dnf(self, coef, exp_CNF3, exp_DNF=None, exp_CNF=None):
        #exp_CNF3 = str(coef)
        with open(self.path_save_exp + 'table_outputblock_' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + ".txt", 'w') as f:
            f.write(str(exp_CNF3))
        if exp_CNF is not None:
            with open(self.path_save_exp + 'CNF_expression_block' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + "_sousblock_" + str(self.sousblockici) + ".txt", 'w') as f:
                f.write(str(exp_CNF))
            with open(self.path_save_exp + 'DNF_expression_block' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + "_sousblock_" + str(self.sousblockici) + ".txt", 'w') as f:
                f.write(str(exp_DNF))


    def iterate_over_filter(self, resici, unique):
        coef_default = unique[0]
        unique2 = unique[1:]
        print(coef_default, unique)
        for unq2 in unique2:
            self.for_1_filter(unq2, resici)
            exp_CNF, exp_DNF, exp_CNF3  = self.for_1_filter(unq2, resici)
            self.save_cnf_dnf(unq2, exp_CNF3, exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3

    def for_1_filter(self, unq2, resici):
        answer = resici == unq2
        dfres = pd.DataFrame(answer)
        dfres.columns = ["Filter_" + str(self.filterici) + "_Value_" + str(int(unq2))]
        df2 = pd.concat([self.df, dfres], axis=1)
        # print(df2)
        df2.to_csv(self.path_save_exp + 'Truth_Table_block' +
                   str(self.blockici) + '_filter_' + str(self.filterici) +
                   '_coefdefault_' +
                   str(unq2) + "_sousblock_" + str(self.sousblockici) + '.csv')
        condtion_filter = df2["index"].values[answer].tolist()
        answer_cnf = (1.0 * answer) == 0.
        # condtion_filter_cnf = df2["index"].values[answer_cnf].tolist()
        exp_DNF, exp_CNF = self.get_expresion_methode1(condtion_filter)
        exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3


    def get_expresion_methode1(self, condtion_filter):
        # TODO dont care term
        if self.n == 4:
            w1, x1, y1, v1 = symbols('x_0, x_1, x_2, x_3')
            exp_DNF = SOPform([w1, x1, y1, v1], minterms=condtion_filter)
            exp_CNF = POSform([w1, x1, y1, v1], minterms=condtion_filter)
        elif self.n == 8:
            w1, x1, y1, v1, w2, x2, y2, v2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter)
            exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter)
        elif self.n == 9:
            w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
            exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
        else:
            #TODO
            pass
        return exp_DNF, exp_CNF



class Block_resnet_multihead_general_BN_vf_small_v3(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, T=0.0, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v3, self).__init__()
        self.cpt = 0
        T = 0.0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,4,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = (2), stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=(2), stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    #print(int(in_planes / g), g, in_planes)
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes,  k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    #self.Block_conv4 =nn.AvgPool2d(2)
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1

        self.stride = stride
        groupvf = 4
        #print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)
        if last:
            self.Block_convf = Block_TT(self.cpt * in_planes, self.cpt*in_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T, last=last)  # int(4*in_planes/4))
        else:
            self.Block_convf = Block_TT(self.cpt * in_planes, out_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T, last=last)  # int(4*in_planes/4))


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
        pad = nn.ZeroPad2d((0,1,0,1))
        pad2 = nn.ZeroPad2d((0, 0, 0, 2))
        pad3 = nn.ZeroPad2d((0, 2, 0, 0))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 56) and out1.shape[-1]==57: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #pass
            #out1 = pad21(out1)
            #out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
        elif (x.shape[-1] == 17):  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            #out1 = pad21(out1)
            #out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #x = pad(x)
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]

        #elif (x.shape[-1] == 8) and out1.shape[-1]==8: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #out1 = pad2(out1)
            #out2 = pad3(out2)

        elif (x.shape[-1] == 4) and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out1 = pad2(out1)
            out2 = pad3(out2)
        elif (x.shape[-1] == 5): #and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out4 = pad(out4)
            out3 = pad(out3)

        elif (x.shape[-1] == 17):
            #out1 = pad(out1)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #out3 = pad(out3)
        elif (x.shape[-1] == 9):
            out4 = pad(out4)
            out3 = pad(out3)
        elif (x.shape[-1] == 7):
            out4 = pad(out4)
            out3 = pad(out3)
            #out1 = pad2(out1)
            #out2 = pad2(out2)
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9) or (x.shape[-1] == 20) or (self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            pass
            #out1 = out1[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        outff = self.Block_convf(outf)
        return outff #torch.cat((outff, out4 ), axis = 1)





class Block_resnet_multihead_general_BN_vf_small_v2(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v2, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,2,4,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 4, stride=stride, padding=2,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=3, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride ==1:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        if stride == 1:
            groupvf = 4
        else:
            groupvf = 3
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
            pass
            #out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, x.shape)
        if (self.stride==1 and x.shape[-1]==7) or (x.shape[-1] == 10) or (self.stride==1 and x.shape[-1]==8) or (self.stride==1 and x.shape[-1]==14) or (self.stride==1 and x.shape[-1]==9) or (self.stride==1 and x.shape[-1]==11)  or (self.stride==1 and x.shape[-1]==12) or (self.stride==1 and x.shape[-1]==20) or (self.stride==1 and x.shape[-1]==18) or (self.stride==1 and x.shape[-1]==16): #or (self.stride == 2 and x.shape[-1] == 13): #or (self.stride == 2 and x.shape[-1] == 13):
            #out1 = self.pad0(out1)
            #out4 = self.pad0(out4)
            out1 = out1[:,:,:-1,:-1]
            out3 = out3[:,:,:-1,:-1]
            #if (x.shape[-1] == 10):
            #    out4 = out4[:, :, :-1, :-1]
        elif (x.shape[-1] == 8) or (x.shape[-1] == 14)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            out1 = out1[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]
            #out4 = out4[:, :, :-1, :-1]
            #out1 = self.pad0(out1)
        #print(out1.shape, out2.shape, out3.shape, x.shape)
        if self.stride == 1:
            outf = torch.cat((out1, out2, out3, out4), axis=1)
        else:
            outf = torch.cat((out1, out2, out3), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)


class Block_resnet_multihead_general_BN_vf(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 7, stride=stride, padding=3,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf = 8
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
        #out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        print(out1.shape, out3.shape, out4.shape, x.shape)
        if (self.stride==1 and x.shape[-1]==7) or (x.shape[-1] == 10) or (self.stride==1 and x.shape[-1]==8) or (self.stride==1 and x.shape[-1]==14) or (self.stride==1 and x.shape[-1]==9) or (self.stride==1 and x.shape[-1]==11)  or (self.stride==1 and x.shape[-1]==12) or (self.stride==1 and x.shape[-1]==20) or (self.stride==1 and x.shape[-1]==18) or (self.stride==1 and x.shape[-1]==16): #or (self.stride == 2 and x.shape[-1] == 13): #or (self.stride == 2 and x.shape[-1] == 13):
            pass
            out1 = self.pad0(out1)
            out4 = self.pad0(out4)
            #out2 = out2[:,:,:-1,:-1]
            out3 = out3[:,:,:-1,:-1]
        elif (x.shape[-1] == 8) or (x.shape[-1] == 14)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            #out2 = out2[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]
            out4 = out4[:, :, :-1, :-1]
            out1 = self.pad0(out1)
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)

        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class Block_resnet_multihead_general_BN_vf_pad(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_pad, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 7, stride=stride, padding=3,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        #print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf = 8
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (self.stride==1 and x.shape[-1]==8) or (self.stride==1 and x.shape[-1]==14) or (self.stride==1 and x.shape[-1]==9) or (self.stride==1 and x.shape[-1]==11)  or (self.stride==1 and x.shape[-1]==12) or (self.stride==1 and x.shape[-1]==20) or (self.stride==1 and x.shape[-1]==18) or (self.stride==1 and x.shape[-1]==16): #or (self.stride == 2 and x.shape[-1] == 13): #or (self.stride == 2 and x.shape[-1] == 13):
            out1 = self.pad0(out1)
            out4 = self.pad0(out4)
            #out2 = out2[:,:,:-1,:-1]
            #out3 = out3[:,:,:-1,:-1]
        elif (x.shape[-1] == 18) or (x.shape[-1] == 14)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out4 = out4[:, :, :-1, :-1]
            out1 = self.pad0(out1)
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)

        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class Block_resnet_multihead_general_BN_vf_17l(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_17l, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 7, stride=stride, padding=3,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        # print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf = 8
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if self.stride==1 or (self.stride == 2 and x.shape[-1] == 16) or (self.stride == 2 and x.shape[-1] == 13):
            out2 = out2[:,:,:-1,:-1]
            out3 = out3[:,:,:-1,:-1]
            out4 = out4[:, :, :-1, :-1]
        elif (self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            out2 = out2[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]
            out4 = out4[:, :, :-1, :-1]

        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class Block_resnet_multihead_general_BN_vf_7x(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_7x, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 7, stride=stride, padding=3,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        # print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf =8
        self.Block_convf = Block_resnet_BN(2 * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(2* in_planes / groupvf))  # int(4*in_planes/4))

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
        #out3 = self.Block_conv3(x)
        #out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if self.stride==1 or (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
            pass
            #out2 = out2[:,:,:-1,:-1]
            #out3 = out3[:,:,:-1,:-1]
        elif (self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            out4 = out4[:, :, :-1, :-1]

        outf = torch.cat((out1, out4), axis =1)#((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, 2, int(c / 2), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class Block_resnet_multihead_general_BN_vf_64(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, groups, stride=1, Abit_inter=2, last=False):
        super(Block_resnet_multihead_general_BN_vf_64, self).__init__()
        self.cpt = 0
        self.groups = groups
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k = 8, stride=stride, padding=4,
                                   groupsici=int(in_planes / g))  # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=4, stride=stride,
                                                       padding=2,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=2, stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g))  # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3:
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    self.Block_conv4 =nn.Sequential(nn. ZeroPad2d(1), nn.AvgPool2d(2))
                    self.cpt += 1
        if Abit_inter > 1:
            self.act = activation_quantize_fn2(Abit_inter)
        else:
            self.act = Binarize01Act()
        self.stride = stride
        # print(self.cpt * in_planes, int(self.cpt * in_planes / g2), g2)
        groupvf = 12
        self.Block_convf = Block_resnet_BN(self.cpt * in_planes, out_planes, Abit_inter=Abit_inter, k=2, stride=1,
                                           padding=1, last=True,
                                           groupsici=int(self.cpt * in_planes / groupvf))  # int(4*in_planes/4))

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
            out4 = self.Block_conv4(x)
        else:
            out4 = x
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if self.stride==1 or (self.stride == 2 and x.shape[-1] == 13) or (self.stride == 2 and x.shape[-1] == 13):
            out1 = out1[:, :, :-1, :-1]
            out2 = out2[:,:,:-1,:-1]
            out3 = out3[:,:,:-1,:-1]
        elif (self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            # print(out1.shape, out2.shape, out3.shape, x.shape)
            out1 = out1[:, :, :-1, :-1]
            out2 = out2[:, :, :-1, :-1]
            out3 = out3[:, :, :-1, :-1]
            out4 = out4[:, :, :-1, :-1]

        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.Block_convf(outf)

class TT_general_vf(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p]#, (4 * p, 2)]

        self.layers.append(nn.BatchNorm2d(p))
        self.layers.append(Binarize01Act())
        self.layers.append(nn.MaxPool2d(2))
        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multihead_general_BN_vf_small_v3(in_planes, out_planes, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        #self.layers.append(nn.Linear(fcsize, 10))
        self.layers.append(Classifier_scale(fcsize, 10, 1000))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 224, 224)).shape[1]  # image size: 64x32
        return int(size)

class Polynome_ACT(nn.Module):
    def __init__(self, alpha=0.47, beta=0.50, gamma=0.09):
        super().__init__()


    def forward(self, input):
        out = 0.47 + 0.50 * input  + 0.09 * input**2 #- 1.7e-10 * input**3 #self.alpha * self.h_function((input / self.gamma) + self.beta) - self.alpha * self.h2_function(self.beta)
        return out

class Classifier_scale(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, fcsize, out_planes, inter=1000):
        super(Classifier_scale, self).__init__()

        self.lin1 = nn.Linear(fcsize, inter, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(inter)
        self.lin2 = nn.Linear(inter, 1000, bias=True)
        self.Polynome_ACT = Polynome_ACT(0,0,0)


    def forward(self, x):
        x = self.BN2(self.lin1(x))
        #if self.training:
        #x = F.relu(x)
        x = self.Polynome_ACT(x)
        #else:
        #    x = self.Polynome_ACT(x)
        return self.lin2(x)


class TT_vf_smallv2(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_small_v2(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        #self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_vf_small(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), 2*p, (4 * p, 2), 4*p, (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_small(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)

class TT_vf_64bit(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_64(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_vf_19lv2(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), 2*p, (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_vf_26(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), 2*p, (4 * p, 2),4*p, (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_vf_30(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), 2*p, (4 * p, 2),4*p, (8 * p, 2), 8*p]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)

class TT_vf_17l(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [(2 * p, 2), (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_17l(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)




class TT_vf_unique_7_x(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_7x(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class TT_vf_18l_pad(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=2, padding=3, groups=1, bias=False)]

        cfg = [p, (2 * p, 2), (4 * p, 2), (8 * p, 2)]

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
            self.layers.append(Block_resnet_multihead_general_BN_vf_pad(in_planes, out_planes, self.args.groups,
                                                              Abit_inter = self.args.Abit_inter, stride=stride, last=last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)