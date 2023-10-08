import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.bahead import BaHeadPlus,BaHeadTiny,BaHeadBase,BaHeadBase_
from nets.mobilenetv2 import *



class MbSegDec(nn.Module):
    def __init__(self,num_classes=2, input_size=224, short_cut_in='2',short_cut_out='2',aspp=True,last_c=True,in_c=24, model_type='test1',conv_type = 'normal'):
        '''
        The network structure of patch classification network

        Parameter:
        - num_classes The number of items that need to be classified on the network
        - input_size Input img size
        - short_cut_in Output layer number for short cut path
        - short_cut_out Input layer number for short cut path
        - aspp Determine whether to add an aspp module at the segmentation head
        - conv_type The type of convolutional module used by the network backbone
        
        '''
        super(MbSegDec, self).__init__()
        input_channel = 32
        width_mult=1
        if conv_type == 'normal':
            block = conv_bn
        else:
            block = InvertedResidual

        if model_type == 'test1':
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1], # 256, 256, 32 -> 256, 256, 16
                [6, 24, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
                [6, 32, 2, 2], # 128, 128, 24 -> 64, 64, 32     4
                # c64 stride 2 -> 1
                [6, 64, 2, 2], # 64, 64, 32 -> 32, 32, 64       7
                # c96 stride 1 -> 2
                [6, 96, 2, 2], # 32, 32, 64 -> 32, 32, 96
                [6, 160, 2, 1], # 32, 32, 96 -> 16, 16, 160     14
                [6, 320, 2, 1], # 16, 16, 160 -> 16, 16, 320
            ]

        if model_type == 'test2':
            interverted_residual_setting = [
                # t, c, n, s
                [1, 32, 2, 2], # 256, 256, 32 -> 256, 256, 16
                [6, 64, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
                [6, 128, 2, 2], # 128, 128, 24 -> 64, 64, 32     4
                [6, 256, 2, 2], # 64, 64, 32 -> 32, 32, 64       7
            ]

        input_channel = int(input_channel * width_mult)
        self.out_bn = nn.AdaptiveAvgPool2d(1)
        self.short_cut_in = short_cut_in
        self.short_cut_out = short_cut_out
        if model_type == 'test1':
            self.stem = nn.ModuleList(
                    nn.Sequential(
                        conv_bn(3, input_channel, 2)
                    )
                )
        else:
            self.stem = nn.ModuleList(
                    nn.Sequential(
                        conv_bn_(3, input_channel, 2)
                    )
                )

        self.features = nn.ModuleDict()

        self.seg_head = BaHeadBase_(in_c = in_c,aspp = True)

        for idx,(t, c, n, s) in enumerate(interverted_residual_setting):
            num = idx+1
            output_channel = int(c * width_mult)
            blocks = []
            for i in range(n):
                if i == 0 and num != int(short_cut_out)+1:
                    blocks.append(block(input_channel, output_channel, s, t))
                elif i == 0 and num == int(short_cut_out)+1:
                    blocks.append(block(input_channel+2, output_channel, s, t))
                else:
                    blocks.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
            self.features[str(num)] = nn.Sequential(*blocks)

        self.last_c = last_c
        if self.last_c == True:
            self.last_channel = nn.ModuleList(
                                        nn.Sequential(
                                            conv_1x1_bn(input_channel,1280)
                                        )
                                    )
            self.head = nn.Linear(1280,num_classes)
        else:
            self.head = nn.Linear(512,num_classes)


        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self,input):
        row_, col_ = input.size()[2],input.size()[3]
        data = self.stem[0](input)
        for key in list(self.features.keys()):
            data= self.features[key](data)
            if key == self.short_cut_in:
                seg_output = self.seg_head(data)
            
            if key == self.short_cut_out:
                row, col = data.size()[2],data.size()[3]
                seg_output = F.interpolate(seg_output, size=(row, col), mode='bilinear', align_corners=True)
                data = torch.cat([data,seg_output],dim=1)
        if self.last_c == True:
            data = self.last_channel[0](data)
            data = self.avg_pool(data)
        else:
            data_0 = self.avg_pool(data)
            data_1 = self.max_pool(data)
            data = torch.cat([data_0,data_1],dim=1)

        data = torch.flatten(data, 1)
        cls_out = self.head(data)
        seg_output = F.interpolate(seg_output, size=(row_, col_), mode='bilinear', align_corners=True)

        return seg_output,cls_out


if __name__ == "__main__":
    model = MbSegDec()