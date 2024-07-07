import torch
import torch.nn as nn
from util import ModelUtils as modelUtils
import torch.nn.functional as F




class NCA(torch.nn.Module):
    def __init__(self, parameterization):
        super(NCA, self).__init__()
        self.input_dimension = parameterization.get("in_dim", 16)+87
        self.neurons = parameterization.get("neu_num", 32)
        self.n_hidden_layers = parameterization.get("hid_lay_num", 3)
        self.output_dimension = self.input_dimension - 3
        self.kern_size = parameterization.get("kernel_size", 3)
        self.pad_size = self.kern_size//2
        self.pad_mode = parameterization.get("padding_mode", 'zeros')
        # define dilated Conv layers
        self.n_dicon_lay = parameterization.get("dic_lay_num", 0)
        self.dicon_neurons = parameterization.get("dic_neu_num", 0)
        self.dr = parameterization.get("drop", 0.0)
        if self.n_dicon_lay != 0:
            self.dic_input_layer = nn.Conv3d(self.input_dimension, self.dicon_neurons, kernel_size=3,
                                             padding=1+self.kern_size//2, padding_mode=self.pad_mode, dilation=1+self.kern_size//2,
                                             bias=False)
            self.dic_c_con = nn.Conv3d(self.neurons + self.dicon_neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode,  bias=False)
            if self.n_dicon_lay > 1:
                self.dic_layers = nn.ModuleList(
                    [nn.Conv3d(self.dicon_neurons + self.neurons, self.dicon_neurons, kernel_size=3,
                                             padding=1+self.kern_size//2, padding_mode=self.pad_mode, dilation=1+self.kern_size//2,bias=False)
                     for _ in range(self.n_dicon_lay - 1)])
            else:
                self.dic_layers = nn.ModuleList([])

        self.input_layer = nn.Conv3d(self.input_dimension, self.neurons, kernel_size=self.kern_size,
                                     padding=self.pad_size,
                                     padding_mode=self.pad_mode, bias=False)
        if self.n_dicon_lay > 1:
            self.hid_with_dilay = nn.ModuleList(
                [nn.Conv3d(self.neurons + self.dicon_neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode, bias=False)
                 for _ in range(self.n_dicon_lay-1)])
            self.hidden_layers = nn.ModuleList(
                [nn.Conv3d(self.neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode, bias=False)
                 for _ in range(self.n_hidden_layers-self.n_dicon_lay)])
        else:
            self.hid_with_dilay = nn.ModuleList([])
            self.hidden_layers = nn.ModuleList(
                [nn.Conv3d(self.neurons+i*64, self.neurons+(i+1)*64, kernel_size=self.kern_size, padding=self.pad_size, padding_mode=self.pad_mode, bias=False)
                 for i in range(self.n_hidden_layers)])

        if self.n_hidden_layers == self.n_dicon_lay-1:
            self.output_layer = nn.Conv3d(self.neurons+self.dicon_neurons, self.output_dimension, kernel_size=1, bias=False)
        else:
            self.output_layer = nn.Conv3d(self.neurons*2+(self.n_hidden_layers)*64, self.output_dimension, kernel_size=1,
                                          bias=False)

        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=self.dr)



    def forward(self, x):

        x[:,:91,...] = x[:,:91, ...] * (x[:,91:92,...]>0)* (x[:,92:93,...]>0)
        x[:,94:,...] = x[:,94:, ...] * (x[:,91:92,...]>0)* (x[:,92:93,...]>0)

        input_x = x
        live_mask = modelUtils.get_living_mask(input_x)
        solid_mask = modelUtils.non_liquid_mask(input_x)

        if self.n_dicon_lay != 0:
            x2 = x
            x2 = self.activation(self.dic_input_layer(x2))
            x = self.activation(self.input_layer(x))
            x = torch.cat([x, x2], axis=1)
        else:
            x = self.activation(self.input_layer(x))
        x2 = torch.clone(x)
        if self.n_dicon_lay > 1:
            for k, (l,dl) in enumerate(zip(self.hid_with_dilay, self.dic_layers)):
                x2 = x
                x2 = self.activation(dl(x2))
                x = self.activation(l(x))
                x = torch.cat([x, x2], axis=1)
            x = self.activation(self.dic_c_con(x))
            for j in range(self.n_hidden_layers-self.n_dicon_lay):
                l = self.hidden_layers[j]
                x = self.activation(l(x))
        else:
            if self.n_dicon_lay != 0:
                x = self.activation(self.dic_c_con(x))
            for k, l in enumerate(self.hidden_layers):
                x = self.activation(l(x))

        if self.dr!= 0.0:
            x = self.dropout(x)
        y1 = self.output_layer(torch.cat([x, x2], axis=1))

        #class_change
        y = torch.cat(
            [y1[:, :91, ...], input_x[:, 6:9, ...] * 0.0, y1[:, 91:, ...]],
            axis=1)

        output_x = input_x + y * live_mask * solid_mask * (input_x[:,91:92,...]>0)* (input_x[:,92:93,...]>0)

        return output_x

    def initialize_weights(self):
        torch.nn.init.constant_(self.output_layer.weight.data, 0.0)
  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()



