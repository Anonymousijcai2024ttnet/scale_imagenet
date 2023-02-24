import copy
from collections import OrderedDict
import os


from .model_utils.netbin import SeqBinModelHelper, BinLinearPos, g_weight_binarizer, \
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
from torch.autograd import Function
import weakref
import functools
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from sympy import symbols, SOPform, POSform, simplify_logic

g_channel_scale = 1
g_bingrad_soft_tanh_scale = 1

class Polynome_ACT(nn.Module):
    def __init__(self, alpha=0.47, beta=0.50, gamma=0.09):
        super().__init__()


    def forward(self, input):
        out = 0.47 + 0.50 * input  + 0.09 * input[:,:160]**2 #- 1.7e-10 * input**3 #self.alpha * self.h_function((input / self.gamma) + self.beta) - self.alpha * self.h2_function(self.beta)
        return out


def scale_channels(x: int):
    return max(int(round(x * g_channel_scale)), 1)


class AbstractTensor:
    """tensor object for abstract interpretation (here we use interval
    arithmetic)
    """
    __slots__ = ['vmin', 'vmax', 'loss']

    loss_layer_decay = 1
    """decay of loss of previous layer compared to current layer"""

    def __init__(self, vmin: torch.Tensor, vmax: torch.Tensor,
                 loss: torch.Tensor):
        assert vmin.shape == vmax.shape and loss.numel() == 1
        self.vmin = vmin
        self.vmax = vmax
        self.loss = loss

    def apply_linear(self, w, func):
        """apply a linear function ``func(self, w)`` by decomposing ``w`` into
        positive and negative parts"""
        wpos = F.relu(w)
        wneg = w - wpos
        vmin_new = func(self.vmin, wpos) + func(self.vmax, wneg)
        vmax_new = func(self.vmax, wpos) + func(self.vmin, wneg)
        return AbstractTensor(torch.min(vmin_new, vmax_new),
                              torch.max(vmin_new, vmax_new),
                              self.loss)

    def apply_elemwise_mono(self, func):
        """apply a non-decreasing monotonic function on the values"""
        return AbstractTensor(func(self.vmin), func(self.vmax), self.loss)

    @property
    def ndim(self):
        return self.vmin.ndim

    @property
    def shape(self):
        return self.vmin.shape

    def size(self, dim):
        return self.vmin.size(dim)

    def view(self, *shape):
        return AbstractTensor(self.vmin.view(shape), self.vmax.view(shape),
                              self.loss)


class MultiSampleTensor:
    """tensor object that contains multiple samples within the perturbation
    bound near the natural image

    :var data: a tensor ``(K * N, C, H, W)`` where ``data[0:N]`` is the natural
        image, and data[N::N] are perturbations
    """
    __slots__ = ['k', 'data', 'loss']

    loss_layer_decay = 1
    """decay of loss of previous layer compared to current layer"""

    def __init__(self, k: int, data: torch.Tensor, loss: torch.Tensor = 0):
        self.k = k
        self.data = data
        self.loss = loss
        assert data.shape[0] % k == 0

    @classmethod
    def from_squeeze(cls, data: torch.Tensor, loss=0):
        """``data`` should be in ``(K, N, ...)`` format"""
        k, n, *other = data.shape
        return cls(k, data.view(k * n, *other), loss)

    def apply_batch(self, func, loss=None):
        """apply a batch function"""
        if loss is None:
            loss = self.loss
        return MultiSampleTensor(self.k, func(self.data), loss)

    def as_expanded_tensor(self) -> torch.Tensor:
        """expand the first dimension to ``(K, N)``"""
        kn, *other = self.shape
        k = self.k
        n = kn // k
        return self.data.view(k, n, *other)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim):
        return self.data.size(dim)

    def view(self, *shape):
        assert shape[0] == self.shape[0]
        return MultiSampleTensor(self.k, self.data.view(shape), self.loss)


class BinaryActivation(nn.Module):
    '''
        learnable distance and center for activation
    '''

    def __init__(self):
        super(BinaryActivation, self).__init__()
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.beta_a = nn.Parameter(torch.tensor(0.0))

    def gradient_approx(self, x):
        '''
            gradient approximation
            (https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
        '''
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

    def forward(self, x):
        x = (x - self.beta_a) / self.alpha_a
        x = self.gradient_approx(x)
        return self.alpha_a * (x + self.beta_a)

class Binarize01Act(nn.Module):

    class Fn(Function):
        @staticmethod
        def forward(ctx, inp, T, scale=None):
            """:param scale: scale for gradient computing"""
            if scale is None:
                ctx.save_for_backward(inp)
            else:
                ctx.save_for_backward(inp, scale)
            all_ones = 1.0*(inp >= T/2)
            maks = 1.0*(inp < T/2) - 1.0*( inp < -T/2)
            #print(torch.sum(maks) / (maks.shape[0]*maks.shape[1]*maks.shape[2]*maks.shape[3]))
            random = torch.randint_like(inp, 2).to(inp.dtype)
            res = all_ones + maks*random

            return res.to(inp.dtype)

        @staticmethod
        def backward(ctx, g_out):
            if len(ctx.saved_tensors) == 2:
                inp, scale = ctx.saved_tensors
            else:
                inp, = ctx.saved_tensors
                scale = 1

            if g_bingrad_soft_tanh_scale is not None:
                scale = scale * g_bingrad_soft_tanh_scale
                tanh = torch.tanh_(inp * scale)
                return (1 - tanh.mul_(tanh)).mul_(g_out), None, None

            # grad as sign(hardtanh(x))
            g_self = (inp.abs() <= 1).to(g_out.dtype)
            return g_self.mul_(g_out), None, None

    def __init__(self, T=0, T_test = 0,grad_scale=1):
        super().__init__()
        self.T = T
        self.T_test = T_test
        print(self.T)
        self.register_buffer(
            'grad_scale',
            torch.tensor(float(grad_scale), dtype=torch.float32))


    def forward(self, x):
        grad_scale = getattr(self, 'grad_scale', None)
        #if self.T == 0:
        #if x.requires_grad == True:
        f = lambda x: self.Fn.apply(x, self.T, grad_scale)#thr_bin_act=self.thr_bin_act)
        # elif x.requires_grad != True and self.T !=0:
        #     f = lambda x: self.Fn.apply(x, self.T+self.T_test, grad_scale)#thr_bin_act=self.thr_bin_act)
        #else:
        #    f = lambda x: self.Fn.apply(x, self.T, grad_scale)

        def rsloss(x, y):
            return (1 - torch.tanh(1 + x * y)).sum()

        if type(x) is AbstractTensor:
            loss = rsloss(x.vmin, x.vmax)
            loss += x.loss * AbstractTensor.loss_layer_decay
            vmin = f(x.vmin)
            vmax = f(x.vmax)
            return AbstractTensor(vmin, vmax, loss)
        elif type(x) is MultiSampleTensor:
            rv = x.as_expanded_tensor()
            loss = rsloss(rv[-1], rv[-2])
            return x.apply_batch(
                f,
                loss=x.loss * MultiSampleTensor.loss_layer_decay + loss
            )
        else:
            return f(x)


def get_exp_with_y(exp_DNFstr, exp_CNFstr):
    exp_DNFstr, exp_CNFstr = str(exp_DNFstr).replace(" ", ""), str(exp_CNFstr).replace(" ", "")
    masks = exp_DNFstr.split("|")
    clausesnv = []
    for mask in masks:
        # print(mask)
        masknv = mask.replace("&", " | ")
        masknv = masknv.replace("x", "~x")
        masknv = masknv.replace("~~", "")
        masknv = masknv.replace(")", "").replace("(", "")
        masknv = "(" + masknv + ")"
        masknv = masknv.replace("(", "(y | ")
        clausesnv.append(masknv)
        # print(masknv)
    clauses = exp_CNFstr.split("&")
    for clause in clauses:
        # print(clause)
        clausenv = clause.replace("|", " | ")
        clausenv = clausenv.replace(")", "").replace("(", "")
        clausenv = "(" + clausenv + ")"
        clausenv = clausenv.replace(")", " | ~y)")
        clausesnv.append(clausenv)
    exp_CNF3 = " & ".join(clausesnv)

    return exp_CNF3


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
        self.act = Binarize01Act(T=T)

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
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,4,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = 2, stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=2, stride=stride,
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
            self.Block_convf = Block_TT(self.cpt * in_planes, 4*out_planes, k=1, stride=1,
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
        if (x.shape[-1] == 16) and out1.shape[-1]==17: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
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
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
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



class TT_FHE_SMALL(SeqBinModelHelper, nn.Module, ModelHelper):
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
        self.layers = [nn.Conv2d(3, p, kernel_size=5, stride=2, padding=2, groups=1, bias=False)]
        cfg = [p]#, (4*p,2)]
        self.T = 0.0
        self.T_block = 0.0
        self.layers.append(nn.BatchNorm2d(p))
        self.layers.append(Binarize01Act(T=self.T))
        in_planes = p
        last = False
        last_out_planes = cfg[-1] if isinstance(cfg[-1], int) else cfg[-1][0]
        for index_x, x in enumerate(cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if out_planes == last_out_planes:
                last = False
            self.layers.append(Block_resnet_multihead_general_BN_vf_small_v3(in_planes, out_planes, stride=stride, T=self.T_block, last=last))
            in_planes = 4*out_planes
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(Flatten())
        self.features_before_LR = nn.Sequential(*self.layers)
        fcsize = self.linear_input_neurons()
        del self.features_before_LR
        #self.layers.append(Classifier_scale(fcsize, 10, 100))
        #self.layers.append(nn.Linear(fcsize, 100, bias=False))
        #self.layers.append(nn.ReLU(inplace=True))
        #self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(fcsize, 100, bias=False))
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100, 10, bias=False))
        self.layers.append(nn.BatchNorm1d(10))
        self.features = nn.Sequential(*self.layers)



    def linear_input_neurons(self):
        size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        return int(size)


class Classifier_scale_new(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, fcsize, out_planes=10, inter=50, non_lin=960):
        super(Classifier_scale_new, self).__init__()
        self.non_lin = non_lin
        self.lin1 = nn.Linear(non_lin, inter, bias=False)
        #self.relu = nn.ReLU(inplace=True)
        self.BN1 = nn.BatchNorm1d(inter)
        self.lin2 = nn.Linear(fcsize-non_lin, inter, bias=False)
        #self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(inter)
        self.linvf = nn.Linear(inter+inter, out_planes, bias=True)
        self.BNvf = nn.BatchNorm1d(out_planes)
        self.poly = Polynome_ACT(alpha=0.47, beta=0.50, gamma=0.09)


    def forward(self, x):
        x1 = x[:, :self.non_lin]
        x2 = x[:, self.non_lin:]
        x1 = self.BN1(self.lin1(x1))
        x2 = self.BN2(self.lin2(x2))
        x1 = self.poly(x1)
        xf = torch.cat((x1,x2), axis=1)
        return self.BNvf(self.linvf(xf))








class Classifier_scale(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, fcsize, out_planes, inter=100):
        super(Classifier_scale, self).__init__()

        self.lin1 = nn.Linear(fcsize, inter, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(inter)
        self.lin2 = nn.Linear(inter, out_planes, bias=True)
        self.Polynome_ACT = Polynome_ACT(0,0,0)


    def forward(self, x):
        x = self.BN2(self.lin1(x))
        #if self.training:
        #x = F.relu(x)
        x = self.Polynome_ACT(x)
        #else:
        #    x = self.Polynome_ACT(x)
        return self.lin2(x)
