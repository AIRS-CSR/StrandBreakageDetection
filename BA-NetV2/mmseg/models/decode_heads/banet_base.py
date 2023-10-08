import torch.nn as nn
import torch
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

        

class se_block(nn.Module):
    def __init__(self, channels, ratio=16):
        '''
        illustrate : 
        - The block of senet

        Input para :
        - channels : input channels
        '''
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        '''
        Illustrate :
        - channel attention

        Input param :
        - in_planes : input channels
        - ratio : ratio
        '''
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        '''
        Illustrate : 
        - spatial attention 
        Input Param :
        - kernel_size
        '''
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class Block(nn.Module):
    def __init__(self,inplanes,planes,k_size=3,stride=1,attention=False,c_a=None,s_a=None,se_a=se_block,dilation=None):
        '''
        Illustrate :
        - This module is used to construct convolutional modules for each parallel branch
        - ccm : conv+bn+relu
        - fam : att+conv+bn+relu
        
        input_para:
        - inplanes : Number of input channels
        - planes : Number of output channels
        - k_size : Convolutional kernel size
        - stride : Convolutional kernel stride
        - attention : Determine whether to use attention
        - c_a : The module of channel attention
        - s_a : The module of spatial attention
        - se_a : Senet
        - dilation : dialation convolutional kernel
        '''
        super(Block,self).__init__()
        
        if dilation is not None:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=k_size, stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=k_size, stride=stride, padding=1, bias=False)
            
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.attetion = attention
        if attention == 'cs_a':
            self.c_a = c_a
            self.s_a = s_a
            self.se_a = None
        if attention == 'se_a':
            self.se_a = se_a(inplanes)
            self.c_a = None
            self.s_a = None
        if attention == 'ses_a':
            self.se_a = se_a(inplanes)
            self.c_a = None
            self.s_a = s_a

        if attention == False:
            self.se_a = None
            self.c_a = None
            self.s_a = None
            
    
    def forward(self,x):
        if self.attetion is not None:
            if self.se_a is not None:
                x = self.se_a(x)
            if self.c_a is not None:
                x = self.c_a(x) * x
            if self.s_a is not None:
                x = self.s_a(x) * x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    

class FuseModule(nn.Module):
    def __init__(self,
                input_branches,
                fuse_branches,
                output_branches,
                input_channels,
                base_channel,
                level_num,
                att,
                fuse,
                dilation,
                dilation_list):
        '''
        illustrate:
        - This module is used to build parallel branches
        - input -> conv -> deconv -> fuse -> output
        
        input_para:
        - input_branches : Branches that require feature extraction
        - fuse_branch : Number of upsampling branches required
        - output_branches : Number of branches in the output feature map
        - input_channel : Number of channels for input feature maps
        - base_channel : Number of basic channels per branch
        - level_num : Number of feature levels
        - att : Setting of Attention Module
        - dilation : Determine the setting for using dialation convolution
        - dilation_list : Set at which branch to use dialation convolution
        '''
        super().__init__()
    
        self.level_num = level_num
        self.branches = nn.ModuleList()
        self.fuse_branches = fuse_branches
        self.fuse = fuse
    
        for i,c in enumerate(input_channels):
            if i <= input_branches-1:
                if dilation != None and i in dilation_list:
                    branch = nn.Sequential(
                        Block(c,base_channel,3,1,att,ChannelAttention(c),SpatialAttention(),dilation=dilation)
                    )
                else:
                    branch = nn.Sequential(
                        Block(c,base_channel,3,1,att,ChannelAttention(c),SpatialAttention(),dilation=None)
                    )
            else:
                branch = nn.Sequential(
                    nn.Identity()
                )
            self.branches.append(branch)

        self.fuse_idx = [i-1 for i in range(level_num) if i <= fuse_branches and i != 0]
    


    def forward(self,x):
        x_conv = [self.branches[i](x[i]) for i in range(self.level_num)]
        x_upsample = []

        for i in range(len(x_conv)):
            if i <= self.fuse_branches and i != 0:
                #print(i)
                if x_conv[i].size(2) != x_conv[i-1].size(2):
                    x_upsample.append(F.interpolate(x_conv[i], size=(x_conv[i-1].size(2),x_conv[i-1].size(3)), mode='bilinear', align_corners=True))
                else:
                    x_upsample.append(x_conv[i])
            else:
                x_upsample.append(x_conv[i])

        outs = []
        for i in range(self.level_num):
            if i in self.fuse_idx:
                if self.fuse == 'concat':
                    outs.append(torch.concat([x_conv[i],x_upsample[i+1]],dim=1))
                else:
                    outs.append(x_conv[i]+x_upsample[i+1])
            else:
                outs.append(x_conv[i])

        return outs
    
    

@HEADS.register_module()
class BaHeadBase(BaseDecodeHead):
    def __init__(
                self,
                layer0=[5,4,5,[16,16,32,32,32],16,5,False],
                layer1=[4,3,5,[32,32,32,32,16],16,5,True],
                layer2=None,
                layer3=None,
                layer4=None,
                
                last_layer_channel = 2,
                channels=16,
                base_channel=16,
                level_num=5,
                aspp=False,
                aspp_branch=-1,
                out_att=False,

                fuse = 'concat',
                dilation =[None,None,None,None,None],
                dilation_list=[],
                last_layer =True,
                num_classes = 2,
                vis_bool = False,
                vis_branch = None,
                loss_decode = dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
                ):
        '''
        illustrate : 
        - The complete architecture of Banetv2
        '''
        super(BaHeadBase, self).__init__(in_channels=32,channels=channels,num_classes=num_classes,loss_decode=loss_decode)
        self.aspp = aspp
        self.level_num = level_num
        self.aspp_branch = aspp_branch
        self.out_att = out_att
        self.fuse = fuse

        self.vis_bool = vis_bool
        self.vis_branch = vis_branch

        self.layer_0 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None

        self.dilation = dilation
        if layer0 != None:
            self.layer_0 = FuseModule(layer0[0],layer0[1],layer0[2],layer0[3],layer0[4],layer0[5],layer0[6],self.fuse,self.dilation[0],dilation_list)
        if layer1 != None:
            self.layer_1 = FuseModule(layer1[0],layer1[1],layer1[2],layer1[3],layer1[4],layer1[5],layer1[6],self.fuse,self.dilation[1],dilation_list)
        if layer2 != None:
            self.layer_2 = FuseModule(layer2[0],layer2[1],layer2[2],layer2[3],layer2[4],layer2[5],layer2[6],self.fuse,self.dilation[2],dilation_list)
        if layer3 != None:
            self.layer_3 = FuseModule(layer3[0],layer3[1],layer3[2],layer3[3],layer3[4],layer3[5],layer3[6],self.fuse,self.dilation[3],dilation_list)
        if layer4 != None:
            self.layer_4 = FuseModule(layer4[0],layer4[1],layer4[2],layer4[3],layer4[4],layer4[5],layer4[6],self.fuse,self.dilation[4],dilation_list)

        self.last_layer = None
        self.bottleneck = None

        if last_layer == True:
            self.last_layer = nn.ModuleList(
                nn.Conv2d(base_channel,last_layer_channel,1,1) for i in range(self.level_num)
            )

        if self.aspp == True:
            self.aspp_head = ASPP(base_channel,base_channel)
        if self.out_att == True:
            self.ca = ChannelAttention(base_channel)
            self.sa = SpatialAttention()

        self.relu = nn.ReLU()
        

    def forward(self,x):
        w,h = x[0].size()[2],x[0].size()[3]

        if self.layer_0 != None:
            out = self.layer_0(x)
        if self.layer_1 != None:
            out = self.layer_1(out)
        if self.layer_2 != None:
            out = self.layer_2(out)
        if self.layer_3 != None:
            out = self.layer_3(out)
        if self.layer_4 != None:
            out = self.layer_4(out)

        if self.aspp == True:
            out[self.aspp_branch] = self.aspp_head(out[self.aspp_branch])

        for i in range(len(out)):
            if self.last_layer is not None:
                out[i] = self.last_layer[i](out[i])
            if out[i].size()[2]!=w and out[i].size()[3]!=h:
                out[i] = F.interpolate(out[i], size=(w,h), mode='bilinear', align_corners=True)
        
        outs = torch.cat(out,dim=1)
        if self.bottleneck is not None:
            out = self.bottleneck(out)

        if self.out_att == True:
            outs = self.ca(outs) * outs
            outs = self.sa(outs) * outs
        outs = self.cls_seg(outs)

        if self.vis_bool == False:
            return outs
        if self.vis_bool == True:
            print(self.vis_branch)
            return out[self.vis_branch]

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses



if __name__ == '__main__':
    #data = [torch.randn([4,16,160,160]) for i in range(5)]
    data = [
        torch.randn([4,16,640,640]),
        torch.randn([4,16,320,320]),
        torch.randn([4,32,160,160]),
        torch.randn([4,32,80,80]),
        torch.randn([4,32,40,40]),
    ]
    net = BaHeadPlus()

    net(data)
