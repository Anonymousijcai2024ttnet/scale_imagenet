from collections import OrderedDict
import os
from .model_utils.netbin import SeqBinModelHelper, BinLinearPos, Binarize01Act, g_weight_binarizer, activation_quantize_fn2, \
    BinConv2d, g_weight_binarizer3, setattr_inplace, BatchNormStatsCallbak, g_use_scalar_scale_last_layer, \
    InputQuantizer
from .model_utils.utils import ModelHelper, Flatten
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F




class model_general(SeqBinModelHelper, nn.Module, ModelHelper):

    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self, args):
        super().__init__()
        self._setup_network(args)



    def _setup_network(self, args):
        self.make_small_network(self, args)

    @classmethod
    def make_small_network(
            cls, self,args):
        nclass = 10
        lin = BinLinearPos
        wb = g_weight_binarizer
        wb3 = g_weight_binarizer3
        act = Binarize01Act
        act3 = activation_quantize_fn2
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'


        liste_fonctions = []#OrderedDict([])
        #Preprocessing
        print(args)
        if args.type_weigths_preprocessing_CNN == "ter":
            liste_fonctions.append(BinConv2d(wb3,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "bin":
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "float":
            liste_fonctions.append(nn.Conv2d(in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2]))
        else:
            raise 'PB'
        liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act())

        # Blocks
        input_channel_here = args.preprocessing_CNN[0]
        for numblock, _ in enumerate(args.filters):
            liste_fonctions.append(nn.Conv2d(input_channel_here,
                                            args.filters[numblock] * args.amplifications[numblock],
                                            args.kernelsizes[numblock],
                                            stride=args.strides[numblock],
                                            padding=args.paddings[numblock],
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock] * args.amplifications[numblock]))
            liste_fonctions.append(nn.ReLU())
            liste_fonctions.append(nn.Conv2d(args.filters[numblock] * args.amplifications[numblock],
                                            args.filters[numblock],
                                            1,
                                            stride=1,
                                            padding=0,
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock]))
            liste_fonctions.append(act())
            input_channel_here = args.filters[numblock]
        #CLASSIFICATION
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR

        if args.type_weigths_final_LR == "ter":
            liste_fonctions.append(lin(wb3, fcsize, nclass))
        elif args.type_weigths_final_LR == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
        elif args.type_weigths_final_LR == "float":
            liste_fonctions.append(nn.Linear(fcsize, nclass))
        self.feature_pos = len(liste_fonctions)
        liste_fonctions.append(setattr_inplace(
            BatchNormStatsCallbak(
                self, nclass,
                use_scalar_scale=g_use_scalar_scale_last_layer),
            'bias_regularizer_coeff', 0)
        )
        self.features = nn.Sequential(*liste_fonctions)

        if args.g_remove_last_bn=="True":
            self.features = self.features[:-1]
            self.feature_pos = None



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        else:
            raise "PB"
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader


    def linear_input_neurons(self, args):
        if args.dataset =="CIFAR10":
            size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        elif args.dataset =="MNIST":
            size = self.features_before_LR(torch.rand(1, 1, 28, 28)).shape[1]  # image size: 64x32
        else:
            raise 'PB'

        return int(size)






class Block_resnet(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, k=3, a=8, padding=1, stride=1, groupsici = 1, last=False):
        super(Block_resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, a*in_planes, kernel_size=k,
                               stride=stride, padding=padding, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(a*in_planes)
        self.conv2 = nn.Conv2d(a*in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride =stride
        self.act = activation_quantize_fn2()#Binarize01Act()
        self.last = last



    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        if self.last:
            out = self.bn2(self.conv2(out))
        else:
            out = self.bn2(self.act(self.conv2(out)))
        #out += self.act(self.shortcut(x))
        out = F.gelu(out)
        return out


class mobilenet_v1_TT(SeqBinModelHelper, nn.Module, ModelHelper):

    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self, args):
        super().__init__()
        self._setup_network(args)
        n = 4
        t = 4
        p = n * t
        cfg = [p, (2 * p, 2), 2 * p, (4 * p, 2), 4 * p, (8 * p, 2), 8 * p, 8 * p, 8 * p, 8 * p, 8 * p, (16 * p, 2),
               16 * p]
        layers = [nn.Conv2d(3, p, kernel_size=7, stride=1, padding=3, groups=1, bias=False),
                  nn.BatchNorm2d(p),
                  Binarize01Act()
                  ]
        in_planes=p
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block_resnet(in_planes, out_planes, stride = stride))
            if index_x !=len(cfg) -1:
                layers.append(Binarize01Act())
            in_planes = out_planes
        layers.append(nn.AvgPool2d(2))
        layers.append(Flatten())
        layers.append(nn.Linear(16*p, 10))
        self.features = nn.Sequential(*layers)


    def _setup_network(self, args):
        self.make_small_network(self, args)

    @classmethod
    def make_small_network(
            cls, self,args):
        pass



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        else:
            raise "PB"
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader




class Block_resnet_multiheadsmall_n_samll64(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, last = False):
        super(Block_resnet_multiheadsmall_n_samll64, self).__init__()
        
        self.conv1 = Block_resnet(in_planes, in_planes, k = 4, stride = stride, padding=1, groupsici = int(in_planes/2))#int(in_planes/1))
        #self.conv2 = Block_resnet(in_planes, in_planes, k = 3, stride = stride, padding=1, groupsici = int(in_planes/2))#int(in_planes/2))
        self.conv3 = Block_resnet(in_planes, in_planes, k = 2, stride = stride, padding=0, groupsici = int(in_planes/8))#int(in_planes/4))
        #self.conv4 = Block_resnet(in_planes, in_planes, k = 1, stride = stride, padding=0, groupsici = int(in_planes/16))#int(in_planes/4))
        #self.conv4 = nn.MaxPool2d(2)
        self.act = activation_quantize_fn2()#Binarize01Act()
        #print(4*in_planes, out_planes, int(4*in_planes/6))
        self.convf = Block_resnet(2*in_planes, out_planes, k=2, stride=1, padding=1, last=True, groupsici= int(2*in_planes/8))#int(4*in_planes/4))
        self.stride = stride
        self.last = last

    def forward(self, x):
        if self.stride ==2 and x.shape[-1]!=13 and x.shape[-1]!=9 and x.shape[-1]!=5:
            #out4 = self.conv4(x)#[:,:,:-1,:-1]
            out1 = self.conv1(x)#[:,:,:-1,:-1]
            out3 = self.conv3(x)#[:,:,:-1,:-1]
            #out2 = self.conv2(x)#[:,:,:-1,:-1]
        elif self.stride ==2 and x.shape[-1]==13:
            #print(ok)
            #out4 = (self.conv4(x))#[:,:,:-1,:-1]
            out1 = (self.conv1(x))#[:,:,:-1,:-1]
            out3 = (self.conv3(x))#[:,:,:-1,:-1]
            #out2 = (self.conv2(x))[:,:,:-1,:-1]
        elif self.stride ==2 and x.shape[-1]==9:
            #print(ok)
            #out4 = (self.conv4(x))#[:,:,:-1,:-1]
            out1 = (self.conv1(x))#[:,:,:-1,:-1]
            out3 = (self.conv3(x))#[:,:,:-1,:-1]
            #out2 = (self.conv2(x))[:,:,:-1,:-1]
        elif self.stride ==2 and x.shape[-1]==5:
            #print(ok)
            #out4 = (self.conv4(x))#[:,:,:-1,:-1]
            out1 = (self.conv1(x))#[:,:,:-1,:-1]
            out3 = (self.conv3(x))#[:,:,:-1,:-1]
            #out2 = (self.conv2(x))[:,:,:-1,:-1]
        else:
            #out4 = x[:,:,:-1,:-1] #self.pad2(x)
            out1 = (self.conv1(x))
            out3 = (self.conv3(x))
            #out2 = (self.conv2(x))[:,:,:-1,:-1]
        #shuffle
        #print(out3.shape, out4.shape)
        #print(out1.shape, out2.shape, x.shape)
        outf = torch.cat(( out1, out3), axis = 1)
        #print(outf.shape)
        n, c, w, h = outf.shape
        outf = outf.view(n, 2, int(c/2), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        return self.convf(outf)
        #if self.last:
        #    return F.gelu(self.convf(outf))
        #else:
        #    return self.act(self.convf(outf))



class mobilenet_v2_TT_32_64(SeqBinModelHelper, nn.Module, ModelHelper):

    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self):
        super().__init__()
        self._setup_network()

    def _setup_network(self):
        self.make_small_network(self)

    @classmethod
    def make_small_network(
            cls, self):
        n = 8
        t = 8
        p = n * t
        cfg = [(2 * p, 2), 2 * p, (4 * p, 2), 4 * p, (8 * p, 2), 8 * p, (16 * p, 2),
               16 * p]
        self.layers = [nn.Conv2d(3, p, kernel_size=7, stride=1, padding=3, groups=1, bias=False),
                       activation_quantize_fn2(),
                       nn.BatchNorm2d(p),
                       ]
        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = True
            self.layers.append(Block_resnet_multiheadsmall_n_samll64(in_planes, out_planes, stride=stride, last = last))
            in_planes = out_planes
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        #self.layers.append(nn.Linear(fcsize, 512))
        #self.layers.append(nn.BatchNorm1d(512))
        #self.layers.append(nn.GELU())
        #self.layers.append(nn.Linear(512, 512))
        #self.layers.append(nn.BatchNorm1d(512))
        #self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(fcsize, 10))
        self.features = nn.Sequential(*self.layers)



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        else:
            raise "PB"
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader

    def linear_input_neurons(self):
        #if args.dataset =="CIFAR10":
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        #elif args.dataset =="MNIST":
        #    size = self.features_before_LR(torch.rand(1, 1, 28, 28)).shape[1]  # image size: 64x32
        #else:
        #    raise 'PB'

        return int(size)

