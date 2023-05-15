"""
@author: luo jie
code for DA_HLGN_MSFF in the paper
"Visual Image Decoding of Brain Activities using a Dual Attention Hierarchical Latent Generative Network with Multi-Scale Feature Fusion"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchFunc import SpectralNorm

class BasicDeconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicDeconv, self).__init__()
        self.deconv = SpectralNorm(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ELU() if relu else None

    def forward(self, x):
        x = self.deconv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.deconv = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ELU() if relu else None

    def forward(self, x):
        x = self.deconv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MKCBlock(torch.nn.Module):
    def __init__(self):
        super(MKCBlock, self).__init__()
        self.basic_conv3 = nn.Sequential(
            BasicConv(3, 64, kernel_size=3, stride=4, padding=1, dilation=1),
        ) 
        self.basic_conv7 = nn.Sequential(
            BasicConv(3, 64, kernel_size=3, stride=4, padding=2, dilation=2),
        )
        self.basic_conv11 = nn.Sequential(
            BasicConv(3, 64, kernel_size=3, stride=4, padding=3, dilation=3),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2_a = nn.Linear(64, 64, bias=False)
        self.fc2_b = nn.Linear(64, 64, bias=False)
        self.fc2_c = nn.Linear(64, 64, bias=False)
    def forward(self, x):
        x1 = self.basic_conv3(x)
        x2 = self.basic_conv7(x)
        x3 = self.basic_conv11(x)
        U = x1 + x2 + x3
        b, c, _, _ = U.size()
        fs = self.avg_pool(U).view(b, c)
        fc1_out = self.fc1(fs)
        out_a = self.fc2_a(fc1_out).unsqueeze_(dim=1)
        out_b = self.fc2_b(fc1_out).unsqueeze_(dim=1)
        out_c = self.fc2_c(fc1_out).unsqueeze_(dim=1)
        out = torch.cat([out_a,out_b,out_c],dim = 1)
        out = self.softmax(out).unsqueeze(-1).unsqueeze(-1)
        w_a = out[:,0]
        w_b = out[:,1]
        w_c = out[:,2]

        out_x1 = x1 * w_a.expand_as(x1)
        out_x2 = x2 * w_b.expand_as(x2)
        out_x3 = x3 * w_c.expand_as(x3)

        return torch.cat([out_x1,out_x2,out_x3],dim = 1)

class Fea2Z(nn.Module):
    def __init__(self, channel, zDim):
        super(Fea2Z, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(channel, 64, kernel_size=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16, zDim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 64 * 16))
        return x

class ImgsEncoder(nn.Module):
    def __init__(self, outDim):
        super(ImgsEncoder, self).__init__()
        self.mkcb = MKCBlock()

        self.fea2Z32 = Fea2Z(192, 1024)
        self.conv32 = nn.Sequential(
            BasicConv(192, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
        )
        self.fea2Z16 = Fea2Z(192, 1024)
        self.conv16 = nn.Sequential(
            BasicConv(192, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
        )
        self.fea2Z8 = Fea2Z(256, 1024)
        self.conv8 = nn.Sequential(
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
        )
        self.fcHigh = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 16, outDim),
            nn.Tanh()
        )

    def forward(self, x):
        catOut = self.mkcb(x)
        out32 = self.conv32(catOut)
        z32 = self.fea2Z32(catOut)
        out16 = self.conv16(out32)
        z16 = self.fea2Z16(out32)
        out8 = self.conv8(out16)
        z8 = self.fea2Z8(out16)
        zhigh = self.fcHigh(out8.view(-1, 256 * 16))
        return zhigh, z8, z16, z32


class ChanelSpace_Attn(nn.Module):
    def __init__(self,in_dim):
        super(ChanelSpace_Attn,self).__init__()
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.out_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, in_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        m_batchsize,C,w ,h = x.size()
        location_num = h * w
        downsampled_num = location_num // 4

        proj_query = self.query_conv(x).view(m_batchsize,-1,location_num).permute(0,2,1)
        proj_key = self.key_conv(x)
        proj_key=F.max_pool2d(proj_key,2).view(m_batchsize,-1,downsampled_num)
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy)

        proj_value = self.value_conv(x)
        proj_value=F.max_pool2d(proj_value,2).view(m_batchsize,-1,downsampled_num)
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = self.out_conv(out.view(m_batchsize,C//2,w,h))
        y = self.avg_pool(x).view(m_batchsize, C)
        y = self.fc(y).view(m_batchsize, C, 1, 1)
        out = self.gamma*out + x * y.expand_as(x)
        return out


class MultiScaleFusion(nn.Module):
    def __init__(self, out_size=64):
        super(MultiScaleFusion, self).__init__()
        self.conv8 = nn.Sequential(
            nn.Upsample((out_size, out_size)),
            BasicConv(128, 32, kernel_size=1),
        )
        self.conv16 = nn.Sequential(
            nn.Upsample((out_size, out_size)),
            BasicConv(128, 32, kernel_size=1),
        )
        self.conv32 = nn.Sequential(
            nn.Upsample((out_size, out_size)),
            BasicConv(64, 32, kernel_size=1),
        )
        self.w8 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.w16 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.w32 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.w64 = nn.Conv2d(32, 32, kernel_size=1, bias=False)

        self.conv64 = nn.Sequential(
            BasicDeconv(32, 32, kernel_size=4, stride=2, padding=1),
            BasicDeconv(32, 3, kernel_size=1,relu=False, bn=False),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, fea8,fea16,fea32,fea64):
        out8 = self.conv8(fea8)
        out16 = self.conv16(fea16)
        out32 = self.conv32(fea32)
        out_w8 = self.w8(out8).unsqueeze(1)
        out_w16 = self.w16(out16).unsqueeze(1)
        out_w32 = self.w32(out32).unsqueeze(1)
        out_w64 = self.w64(fea64).unsqueeze(1)
        out = torch.cat([out_w8, out_w16, out_w32, out_w64], dim=1)
        out = F.sigmoid(out)
        aw8 = out[:,0]
        aw16 = out[:,1]
        aw32 = out[:,2]
        aw64 = out[:,3]
        out8 = out8*aw8
        out16 = out16*aw16
        out32 = out32*aw32
        out64 = fea64*aw64
        outcat = out8+out16+out32+out64
        out = self.conv64(outcat)
        return out

class Z2Fea(nn.Module):
    def __init__(self, channel, size, zDim):
        super(Z2Fea, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(zDim, 64 * 16),
            nn.ELU(),
        )
        self.conv = nn.Sequential(
            nn.Upsample((size, size)),
            BasicDeconv(64, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.conv(x.view(-1, 64, 4, 4))
        return x

class ImgsDecoder(torch.nn.Module):
    def __init__(self, inDim):
        super(ImgsDecoder, self).__init__()
        self.fcHigh = nn.Sequential(
            nn.Linear(inDim, 256 * 16),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            BasicDeconv(256, 256, kernel_size=4, stride=2, padding=1),
            BasicDeconv(256, 128, kernel_size=1, relu=True),
            ChanelSpace_Attn(128),
        )
        self.z2Fea8 = Z2Fea(128, 8, 1024)
        self.conv8 = nn.Sequential(
            BasicDeconv(256, 128, kernel_size=4, stride=2, padding=1),
            BasicDeconv(128, 128, kernel_size=1, relu=True),
            ChanelSpace_Attn(128),
        )
        self.z2Fea16 = Z2Fea(128, 16, 1024)
        self.conv16 = nn.Sequential(
            BasicDeconv(256, 128, kernel_size=4, stride=2, padding=1),
            BasicDeconv(128, 64, kernel_size=1, relu=True),
            ChanelSpace_Attn(64),
        )
        self.z2Fea32 = Z2Fea(64, 32, 1024)
        self.conv32 = nn.Sequential(
            BasicDeconv(128, 64, kernel_size=4, stride=2, padding=1),
            BasicDeconv(64, 32, kernel_size=1),
        )
        self.msfb = MultiScaleFusion(64)

    def forward(self, feasList):
        zhigh, z8, z16, z32 = feasList
        out4 = self.fcHigh(zhigh).view(-1,256,4,4)
        out8 = self.conv4(out4)
        out8Ext = self.z2Fea8(z8)
        out16 = self.conv8(torch.cat((out8, out8Ext),1))
        out16Ext = self.z2Fea16(z16)
        out32 = self.conv16(torch.cat((out16, out16Ext),1))
        out32Ext = self.z2Fea32(z32)
        out64 = self.conv32(torch.cat((out32, out32Ext),1))
        img_out = self.msfb(out8,out16,out32,out64)

        return img_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fea1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralNorm(nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0)),
            nn.BatchNorm2d(64, 0.5), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)),
            nn.BatchNorm2d(128, 0.5), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)),
            nn.BatchNorm2d(256, 0.5), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(kernel_size=6, stride=6))

        self.concat = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 128),
                                    nn.Dropout(0.2), nn.Linear(128, 1),
                                    nn.Sigmoid())

    def forward(self, img):
        fea1_out = self.fea1(img).view(-1, 256)
        validity = self.concat(fea1_out)

        return validity
