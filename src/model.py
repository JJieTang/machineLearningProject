import torch
import torch.nn as nn
import torch.nn.init as init



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
        self.dr = parameterization.get("drop", 0.0)

        self.input_layer = nn.Sequential(
            nn.Conv3d(self.input_dimension, self.neurons, kernel_size=self.kern_size,
                                     padding=self.pad_size,
                                     padding_mode=self.pad_mode, bias=False),
            nn.ReLU())
        self.hidden_layers = nn.Sequential()
        for i in range(self.n_hidden_layers):
            self.hidden_layers.add(
                nn.Conv3d(self.neurons+i*64, self.neurons+(i+1)*64, kernel_size=self.kern_size, 
                            padding=self.pad_size, padding_mode=self.pad_mode, bias=False),
                nn.ReLU()
            )
        self.dropout = nn.Dropout(p=self.dr)
        self.output_layer = nn.Conv3d(self.neurons*2+(self.n_hidden_layers)*64, self.output_dimension, kernel_size=1,
                                bias=False)


    def get_living_mask(self, x):
        alpha = x[:, 90:91, ...]
        max_pool = torch.nn.MaxPool3d(kernel_size=21, stride=1, padding=21//2)
        alpha = max_pool(alpha)
        return alpha > 0.1


    def non_liquid_mask(self, x):
        alpha = x[:, 90:91, ...]
        return alpha < 0.99
    

    def get_alive(self, x):

        live_mask = self.get_living_mask(x)
        solid_mask = self.non_liquid_mask(x)

        return live_mask * solid_mask    
    

    def update(self,x):
        input_x = x

        x = self.input_layer(x)
        x2 = torch.clone(x)
        x = self.hidden_layers(x)

        if self.dr!= 0.0:
            x = self.dropout(x)
        y1 = self.output_layer(torch.cat([x, x2], axis=1))

        #class_change
        y = torch.cat(
            [y1[:, :91, ...], input_x[:, 6:9, ...] * 0.0, y1[:, 91:, ...]],
            axis=1)
  
        return y
    

    def perceive(self, x):
        return (x[:,91:92,...]>0)* (x[:,92:93,...]>0)

    
    def forward(self,x):

        # perceive step
        x[:,:91,...] = x[:,:91, ...] * self.perceive(x[:,:91, ...])
        x[:,94:,...] = x[:,94:, ...] * self.perceive(x[:,94:, ...])

        # get living cells
        pre_life_mask = self.get_alive(x)

        # update step
        dx = self.update(x)
        dx = dx * pre_life_mask * self.perceive(x)

        #add updated value
        new_x = x + dx
        return new_x


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # He initialization: good for ReLU
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

