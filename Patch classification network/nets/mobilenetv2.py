import math
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride,exp=None):
    #print('test')
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_bn_(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 5, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,out_indices=[3,10,13,18],model_type = 'big',pretrained=True):
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if model_type == 'big':
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1], # 256, 256, 32 -> 256, 256, 16
                [6, 24, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
                [6, 32, 3, 2], # 128, 128, 24 -> 64, 64, 32     4
                [6, 64, 4, 1], # 64, 64, 32 -> 32, 32, 64       7
                [6, 96, 3, 1], # 32, 32, 64 -> 32, 32, 96
                [6, 160, 3, 1], # 32, 32, 96 -> 16, 16, 160     14
                [6, 320, 1, 1], # 16, 16, 160 -> 16, 16, 320
            ]
        else:
            interverted_residual_setting = [
                # t, c, n, s
                [6, 16, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
                [6, 32, 2, 2], # 128, 128, 24 -> 64, 64, 32     4
                [6, 32, 2, 1], # 64, 64, 32 -> 32, 32, 64       7
                #[6, 32, 3, 1], # 32, 32, 64 -> 32, 32, 96
                [6, 32, 2, 1], # 32, 32, 96 -> 16, 16, 160     14
            ]


        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.out_indices = out_indices
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        out = []
        for i,blk in enumerate(self.features):
            x = blk(x)
            if i in self.out_indices:
                out.append(x)

        return out

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

    def _load_state_dict(self):
        checkpoint = torch.hub.load_state_dict_from_url(url='https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar')
        model_dict = self.state_dict()
        weight_dict = {key:checkpoint[key] for key in self.state_dict() if key in checkpoint.keys() if self.state_dict()[key].shape == checkpoint[key].shape}
        model_dict.update(weight_dict)

        self.load_state_dict(checkpoint)



def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False,out_indices=[3,10,13,18], **kwargs):
    model = MobileNetV2(n_class=1000, out_indices=out_indices,**kwargs)
    return model

if __name__ == "__main__":
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)
