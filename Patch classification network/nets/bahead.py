import torch.nn as nn
import torch
import torch.nn.functional as F


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
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class BALoss(nn.Module):
    def __init__(self,loss_ce,output_num):
        super(BALoss,self).__init__()
        loss_ce_list = nn.ModuleList()
        for _ in output_num:
            loss_ce_list.append(loss_ce)
    
    def forward(self,x):
        pass
        

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
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
    def __init__(self,inplanes,planes,k_size=3,stride=1,attention=False,c_a=None,s_a=None,se_a=None):
        '''
        说明:
        - 单个卷积块(ccm : conv+bn+relu)与注意力机制(fam : att+conv+bn+relu)
        - x -> attention -> conv -> output
        
        input_para:
        - inplanes 输入通道数
        - planes 输出通道数
        - k_size 卷积核大小
        - stride 卷积核步长
        - attention 是否使用注意力
        - c_a 传入通道注意力机制模块
        - s_a 传入空间注意力机制模块
        '''
        super(Block,self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=k_size, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.attetion = attention
        if attention == True:
            # 通道注意力和senet选一个就行了
            self.c_a = c_a # 通道注意力机制
            self.s_a = s_a # 空间注意力机制
            self.se_a = se_a # senet
            
    
    def forward(self,x):
        if self.attetion == True:
            if self.se_a != None:
                x = self.se_a(x) * x
            if self.c_a != None:
                x = self.c_a(x) * x
            if self.s_a != None:
                x = self.s_a(x) * x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    

class FuseModule(nn.Module):
    def __init__(self,input_branches,fuse_branches,output_branches,input_channels,base_channel,level_num,att):
        '''
        说明:
        - 该模块用于搭建并行分支
        - input -> conv -> deconv -> fuse -> output
        
        input_para:
        - input_branches 需要进行特征提取的分支
        - fuse_branch 上采样分支
        - output_branches 输出分支数
        - input_channel 输入通道数
        - base_channel 基础通道倍率
        - level_num 输入特征图张数
        - att 是否使用注意力机制
        '''
        super().__init__()
    
        self.level_num = level_num
        # conv_braches
        self.branches = nn.ModuleList()
        self.fuse_branches = fuse_branches
    
        for i,c in enumerate(input_channels):
            if i <= input_branches-1:
                branch = nn.Sequential(
                    Block(c,base_channel,3,1,att,ChannelAttention(c),SpatialAttention())
                )
            else:
                branch = nn.Sequential(
                    nn.Identity()
                )
            self.branches.append(branch)
            #w = 0

        self.fuse_idx = [i-1 for i in range(level_num) if i <= fuse_branches and i != 0]
    

    def forward(self,x):
        #for data in x:
        #    print(data.size())
        x_conv = [self.branches[i](x[i]) for i in range(self.level_num)]
        
        x_upsample = []

        for i in range(len(x_conv)):
            if i <= self.fuse_branches and i != 0:
                if x_conv[i].size(2) != x_conv[i-1].size(2):
                    x_upsample.append(F.interpolate(x_conv[i], size=(x_conv[i-1].size(2),x_conv[i-1].size(3)), mode='bilinear', align_corners=True))
                else:
                    x_upsample.append(x_conv[i])
            else:
                x_upsample.append(x_conv[i])

        outs = []
        for i in range(self.level_num):
            if i in self.fuse_idx:
                #print(i)
                outs.append(x_conv[i]+x_upsample[i+1])
            else:
                outs.append(x_conv[i])

        return outs
    
    

class BaHeadPlus(nn.Module):
    def __init__(
                self,
                layer0=[4,3,4,[256,512,1024,2048],128,4,False],
                layer1=[3,2,4,[128,128,128,128],128,4,True],
                layer2=[2,1,4,[128,128,128,128],128,4,True],
                channels=128,
                in_channels=None,
                base_channel=128,
                num_classes=21,

                level_num=4,
                in_indices=4,
                aspp=False,
                aspp_branch=-1,
                out_att=False
                ):
        super(BaHeadPlus, self).__init__()
        self.aspp = aspp
        self.level_num = level_num
        self.in_indices = in_indices
        self.aspp_branch = aspp_branch
        self.out_att = out_att

        self.layer_0 = FuseModule(layer0[0],layer0[1],layer0[2],layer0[3],layer0[4],layer0[5],layer0[6],)
        self.layer_1 = FuseModule(layer1[0],layer1[1],layer1[2],layer1[3],layer1[4],layer1[5],layer1[6],)
        self.layer_2 = FuseModule(layer2[0],layer2[1],layer2[2],layer2[3],layer2[4],layer2[5],layer2[6],)
        #self.layer_3 = FuseModule(layer3[0],layer3[1],layer3[2],layer3[3],layer3[4],layer3[5],layer3[6],)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channel*level_num,base_channel,3,1,padding=1),
            nn.BatchNorm2d(base_channel),
            nn.Conv2d(base_channel,base_channel,3,1,padding=1),
            nn.BatchNorm2d(base_channel),
        )

        if self.aspp == True:
            self.aspp_head = ASPP(base_channel,channels)
        if self.out_att == True:
            self.ca = ChannelAttention(base_channel)
            self.sa = SpatialAttention()

        self.relu = nn.ReLU()
        self.cls_seg = nn.Conv2d(base_channel, num_classes, 1, stride=1)

        self._init_weight()

    def _init_weight(self):
        print('load the init weight...')
        # 初始化权重参数
        # 先通过self.modules()读取网络信息
        # 再通过判断方法的填充nn.BatchNorm3d和nn.Conv3d内的参数
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def forward(self,x):
        #for data in x:
        #    print(data.size())

        w,h = x[0].size()[2],x[0].size()[3]
        #print(len(x))
        if self.in_indices != self.level_num:
            x_upsample = x[0]
            x_upsample = F.interpolate(x_upsample, size=(x_upsample.size(2)*2,x_upsample.size(3)*2), mode='bilinear', align_corners=True)
            x.insert(0,x_upsample)
        #print(len(x))
        #print('test')
        #for data in x:
        #    print(data.size())

        out = self.layer_0(x)

        out = self.layer_1(out)

        out = self.layer_2(out)

        if self.aspp == True:
            #print('test1')
            #print(out[self.aspp_branch].size())
            #print(self.aspp_branch)
            out[self.aspp_branch] = self.aspp_head(out[self.aspp_branch])

        for i in range(len(out)):
            if out[i].size()[2]!=w and out[i].size()[3]!=h:
                out[i] = F.interpolate(out[i], size=(w,h), mode='bilinear', align_corners=True)
                
        out = torch.cat(out,dim=1)

        out = self.bottleneck(out)


        if self.out_att == True:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out_seg = self.cls_seg(out)

        return out_seg



class BaHeadTiny(nn.Module):
    def __init__(
                self,
                aspp_c,
                aspp_out_c,
                shortcut_in_c,
                shoutcut_out_c,
                num_classes,
                fuse_layer_num
                ):
        super(BaHeadTiny, self).__init__()

        self.aspp_head = ASPP(aspp_c,aspp_out_c)
        self.shortcut_conv = Block(shortcut_in_c,shoutcut_out_c)

        # print(aspp_out_c+shoutcut_out_c)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(aspp_out_c+shoutcut_out_c, aspp_out_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(aspp_out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.cls_conv = nn.Conv2d(aspp_out_c, num_classes, 1, stride=1)

    def forward(self,x):
        low_level_feature,high_level_feature = x[0],x[1]
        low_level_out = self.shortcut_conv(low_level_feature)
        high_level_out = self.aspp_head(high_level_feature)
        print()

        high_level_out = F.interpolate(high_level_out, size=(low_level_feature.size(2), low_level_feature.size(3)), mode='bilinear', align_corners=True)
        out = self.cat_conv(torch.cat((high_level_out, low_level_out), dim=1))

        out = self.cls_conv(out)
        return out
        

class BaHeadBase(nn.Module):
    def __init__(
            self,
            in_c,
            aspp = False):
        super(BaHeadBase, self).__init__()
        self.aspp = aspp
        if aspp == True:
            self.aspp_head = ASPP(in_c,in_c)
        #self.cls_conv = nn.Conv2d(in_c, 2, 3, stride=1)
        self.cls_conv = nn.Sequential(
                    nn.Conv2d(in_c, 2, 3, stride=1),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),

                )
    
    def forward(self,x):
        #print('test123123')
        x = x[-1]
        if self.aspp == True:
            #print('test_aspp')
            x = self.aspp_head(x)

        x = self.cls_conv(x)
        return x

class BaHeadBase_(nn.Module):
    def __init__(
            self,
            in_c,
            aspp = False):
        super(BaHeadBase_, self).__init__()
        '''
        input -> 3*3conv c24 -> aspp c24 ->1*1conv c2
        '''
        self.aspp = aspp
        if aspp == True:
            # aspp维持通道数
            self.aspp_head = ASPP(in_c,in_c)
        #self.cls_conv = nn.Conv2d(in_c, 2, 3, stride=1)

        # 3*3卷积维持通道数
        self.cls_conv = nn.Sequential(
                    nn.Conv2d(in_c, in_c, 3, stride=1),
                    nn.BatchNorm2d(in_c),
                    nn.ReLU(inplace=True),
                )

        self.cls_conv_ = nn.Sequential(
                    nn.Conv2d(in_c, 2, 1, stride=1),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),
                )
        
        # 增加1*1卷积神经网络 通道数为2
    
    def forward(self,x):
        #print('test123123')
        seg_output = self.cls_conv(x)
        if self.aspp == True:
            #print('test_aspp')
            seg_output = self.aspp_head(seg_output)
        seg_output = self.cls_conv_(seg_output)
        
        return seg_output