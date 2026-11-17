import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from fvcore.nn import FlopCountAnalysis

from models.mfca import MultiFrequencyChannelAttention
from pvtv2 import pvt_v2_b2
sys.path.append('model')
from EGA import EGA,make_laplace_pyramid


IS2D_models

class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            # nn.Conv2d(out_channel, out_channel,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            # nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        conv2 = self.conv2(x1)
        return conv2 + x1 + x

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class SE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(SE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )

        self.gpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x = x * se

        return x

class MSFCA(nn.Module):

    def __init__(self, in_channels, out_channels, channelAttention_reduce=2):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        self.msal = MultiFrequencyChannelAttention(in_channels=(in_channels // 4) * 2, dct_h=14, dct_w=14, reduction=16)
        self.ca = ChannelAttention(input_channels=(in_channels // 4) * 2,
                                   internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

        self.sig = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SE(in_channels)
    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        inputs = spatial_att * inputs

        split_index = self.C // 2
        part1 = inputs[:, :split_index, :, :]
        part2 = inputs[:, split_index:, :, :]

        # 对第一部分进行通道注意力处理
        channel_att_vec = self.ca(part1)
        part1 = channel_att_vec * part1

        # 对第二部分进行频域通道注意力处理
        freq_att = self.msal(part2)
        part2 = freq_att * part2
        # 合并处理后的特征图
        out = torch.cat([part1, part2], dim=1)

        return out + inputs


class IF(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(IF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, input):
        x_1 = self.avg_pool(input)
        x_2 = self.max_pool(input)
        x1 = self.conv1(x_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x_2).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),
                                      x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
    #    print("out2 shape:", out2.shape)
        out2 = self.sigmoid(out2)

        out = out1 + out2
        out = self.sigmoid(out)
        return out * input

class CSCA_blocks1(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks1, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(out_channel*2, out_channel)
        self.se = SE(out_channel, decay)
        self.sig = nn.Sequential(
            nn.Conv2d(out_channel , out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        point = self.se(point)
        # satt = self.msfca(point)
        # plusatt = point * satt
        # plusatt = torch.cat([point, plusatt], dim=1)
        att = self.sig(point)
        return att


class MSFBNet(nn.Module):
    def __init__(self, n_class=1, decay=2):
        super(MSFBNet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.pool = nn.MaxPool2d(2)

        self.down_conv5 = nn.Sequential(
            Basic_blocks(512, 512, decay),
            IF(512)
        )

        self.up_conv5 = CSCA_blocks1(512, 320, decay)
        self.up_conv4 = CSCA_blocks1(320, 128, decay)
        self.up_conv3 = CSCA_blocks1(128, 64, decay)
        self.up_conv2 = CSCA_blocks1(64, 32, decay)
        self.up_conv1 = CSCA_blocks1(64, 32, decay)

        self.dp5 = nn.Conv2d(512, 1, 1)
        self.dp4 = nn.Conv2d(320, 1, 1)
        self.dp3 = nn.Conv2d(128, 1, 1)
        self.dp2 = nn.Conv2d(64, 1, 1)
        self.dp1 = nn.Conv2d(64, 1, 1)
    #    self.out = nn.Conv2d(32, 1, 3, padding=1)  # 103

        self.conv_x5 = nn.Conv2d(1024, 3, 1)
        self.conv_x4 = nn.Conv2d(512, 3, 1)
        self.conv_x3 = nn.Conv2d(256, 3, 1)
        self.conv_x2 = nn.Conv2d(128, 3, 1)

        self.ega3 = EGA(64)
        self.ega4 = EGA(128)
        self.ega5 = EGA(320)

    def forward(self, inputs):
        grayscale_img = rgb_to_grayscale(inputs)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = inputs.to(device)

        pvt = self.backbone(inputs)

        down1 = pvt[0]
        down2 = pvt[1]
        down3 = pvt[2]
        down4 = pvt[3]
      #  print("down4 shape:", down4.shape)
        center = self.down_conv5(down4)
       # center = down4
        out5 = self.dp5(center)

        out5_1 = F.interpolate(
            out5, (inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)

        #      print("down3 shape:", down3.shape)
        deco4 = self.up_conv5(center, down3)
        #      print("deco4 shape:", deco4.shape)
        out5_2 = F.interpolate(out5, (deco4.shape[2], deco4.shape[3]), mode='bilinear', align_corners=False)

        deco4 = self.ega5(edge_feature, deco4, out5_2)
        # print("deco4 shape:", deco4.shape)
        out4 = self.dp4(deco4)

        out4_1 = F.interpolate(
            out4, (inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)

        deco3 = self.up_conv4(deco4, down2)
        out4_2 = F.interpolate(
            out4, (deco3.shape[2], deco3.shape[3]), mode='bilinear', align_corners=False)

        deco3 = self.ega4(edge_feature, deco3, out4_2)

        out3 = self.dp3(deco3)

        out3_1 = F.interpolate(
            out3, (inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)

        deco2 = self.up_conv3(deco3, down1)
        out3_2 = F.interpolate(
            out3, (deco2.shape[2], deco2.shape[3]), mode='bilinear', align_corners=False)
        deco2 = self.ega3(edge_feature, deco2, out3_2)

        out2 = self.dp2(deco2)

        out2_1 = F.interpolate(
            out2, (inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)

        return out2_1, out3_1, out4_1, out5_1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSFBNet(1, 2)
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    out2_1, out3_1, out4_1, out5_1 = model(input_tensor)

    # summary(model, (3, 256, 256), batch_size=1, device='cpu')
    Flops = FlopCountAnalysis(model, input_tensor)
    print("Total Flops:{}G".format(Flops.total() * 1e-9))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

