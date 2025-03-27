import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class CycleGANModel:
    def __init__(self, weights_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = ResnetGenerator().to(self.device)
        self.load_network(weights_path)
        self.netG.eval()

    def load_network(self, weights_path):
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.netG.load_state_dict(state_dict)

    def preprocess_input(self, input_array):

        if not input_array.flags.writeable:
            input_array = np.copy(input_array)

        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(input_array).float()
        
        # Rearrange dimensions from (H, W, C) to (C, H, W)
        input_tensor = input_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        # Normalize to [-1, 1]
        input_tensor = (input_tensor / 127.5) - 1
        
        return input_tensor

    def postprocess_output(self, output_tensor):
        # Remove batch dimension and move to CPU
        output_tensor = output_tensor.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 255]
        output_tensor = (output_tensor + 1) * 127.5
        
        # Rearrange dimensions from (C, H, W) to (H, W, C)
        output_tensor = output_tensor.permute(1, 2, 0)
        
        # Convert to numpy array and clip values
        output_array = output_tensor.numpy().clip(0, 255).astype(np.uint8)
        
        return output_array

    @torch.no_grad()
    def transform(self, input_array):
        self.netG.eval()
        input_tensor = self.preprocess_input(input_array)
        output_tensor = self.netG(input_tensor.to(self.device))
        return self.postprocess_output(output_tensor)