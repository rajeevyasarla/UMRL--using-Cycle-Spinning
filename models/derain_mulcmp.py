import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable



def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D1(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D1, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf,ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 32
    self.deconv4 = deconv_block(ndf, ndf)
    # 64
    self.deconv3 = deconv_block(ndf, ndf)
    # 128
    self.deconv2 = deconv_block(ndf, ndf)
    # 256
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
    """
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh())
    """
  def forward(self,x):
    out1 = self.conv1(x)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.encode(out4)
    dout5= self.decode(out5)
    dout4= self.deconv4(dout5)
    dout3= self.deconv3(dout4)
    dout2= self.deconv2(dout3)
    dout1= self.deconv1(dout2)
    return dout1

class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output







class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out






class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()




        ############# 256-256  ##############
        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):

        feature=self.feature(x)
        # feature = Variable(feature.data, requires_grad=True)

        feature=self.conv16(feature)
        # print feature.size()

        # feature=Variable(feature.data,requires_grad=True)



        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        # print out.size()

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out


class scale_residue_est(nn.Module):
    def __init__(self):
        super(scale_residue_est, self).__init__()

        self.conv1 = BottleneckBlock(64, 32)
        self.trans_block1 = TransitionBlock3(96, 32)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock3(64, 32)
        self.conv3 = BottleneckBlock(32, 32)
        self.trans_block3 = TransitionBlock3(64, 32)
        self.conv_refin = nn.Conv2d(32, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4 = self.relu((self.conv_refin(x3)))
        residual = self.tanh(self.refine3(x4))

        return residual

class scale_residue_conf(nn.Module):
    def __init__(self):
        super(scale_residue_conf, self).__init__()

        self.conv1 = nn.Conv2d(35,16,3,1,1)#BottleneckBlock(35, 16)
        #self.trans_block1 = TransitionBlock3(51, 8)
        self.conv2 = BottleneckBlock(16, 16)
        self.trans_block2 = TransitionBlock3(32, 16)
        self.conv3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        #x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual


class UMRL(nn.Module):
    def __init__(self):
        super(UMRL, self).__init__()

        self.dense_block1=BottleneckBlock1(3,29)
        self.trans_block1=TransitionBlock1(32,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(32,32)
        self.trans_block2=TransitionBlock1(64,32)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(32,32)
        self.trans_block3=TransitionBlock1(64,32)
        
        self.dense_block3_1=BottleneckBlock1(32,32)
        self.trans_block3_1=TransitionBlock1(64,32)

        self.dense_block3_2=BottleneckBlock1(32,32)
        self.trans_block3_2=TransitionBlock(64,32)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(64,32)
        self.trans_block4=TransitionBlock(96,32)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(67,32)
        self.trans_block5=TransitionBlock(99,32)

        self.dense_block6=BottleneckBlock1(67,16)
        self.trans_block6=TransitionBlock(83,16)
        self.dense_block6_1=BottleneckBlock1(16,16)
        self.trans_block6_1=TransitionBlock3(32,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.res_est = scale_residue_est()
        self.conf_res = scale_residue_conf()



    def forward(self, x,x_256,x_128):
        ## 256x256
        x1_m=self.dense_block1(x)
        x1=self.trans_block1(x1_m)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        #print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)

        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)

        ## Classifier  ##

        x4_in = torch.cat([x3_2, x3], 1)
        x4=(self.dense_block4(x4_in))
        x4=self.trans_block4(x4)
        in_rs128 = torch.cat([x4, x2], 1)
        res_128 = self.res_est(in_rs128)
        conf_128 = self.conf_res(torch.cat([x4, res_128], 1))

        x5_in=torch.cat([x4, x2,res_128*conf_128], 1)

        x5=(self.dense_block5(x5_in))
        x5=self.trans_block5(x5)
        in_rs256 = torch.cat([x5, x1], 1)
        res_256 = self.res_est(in_rs256)
        conf_256 = self.conf_res(torch.cat([x5, res_256], 1))

        x6_in=torch.cat([x5, x1,res_256*conf_256], 1)

        x6=(self.dense_block6(x6_in))
        x6=(self.trans_block6(x6))
        x6=(self.dense_block6_1(x6))
        x6=(self.trans_block6_1(x6))
        
        
        x7=self.relu((self.conv_refin(x6)))
        residual=self.tanh(self.refine3(x7))
        x7_in_2 = torch.cat([x6,x6,residual],1)
        conf_512 = self.conf_res(x7_in_2)
        clean = x - residual
        clean1=self.relu(self.refineclean1(clean))
        clean2=self.tanh(self.refineclean2(clean1))
        clean128 = x_128 - res_128
        clean128 = self.relu(self.refineclean1(clean128))
        clean128 = self.tanh(self.refineclean2(clean128))
        clean256 = x_256 - res_256
        clean256 = self.relu(self.refineclean1(clean256))
        clean256 = self.tanh(self.refineclean2(clean256))

        return residual, clean2, clean128, clean256, conf_128, conf_256, conf_512
        
class Multi_scale2(nn.Module):
    def __init__(self):
        super(Multi_scale2, self).__init__()

        self.dense_block1=BottleneckBlock1(3,29)
        self.trans_block1=TransitionBlock1(32,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(32,32)
        self.trans_block2=TransitionBlock1(64,32)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(32,32)
        self.trans_block3=TransitionBlock1(64,32)
        
        self.dense_block3_1=BottleneckBlock1(32,32)
        self.trans_block3_1=TransitionBlock1(64,32)

        self.dense_block3_2=BottleneckBlock1(32,32)
        self.trans_block3_2=TransitionBlock(64,32)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(64,32)
        self.trans_block4=TransitionBlock(96,32)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(67,32)
        self.trans_block5=TransitionBlock(99,32)

        self.dense_block6=BottleneckBlock1(67,16)
        self.trans_block6=TransitionBlock(83,16)
        self.dense_block6_1=BottleneckBlock1(16,16)
        self.trans_block6_1=TransitionBlock3(32,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.res_est = scale_residue_est()
        self.conf_res = scale_residue_conf()



    def forward(self, x,x_256,x_128):
        ## 256x256
        x1_m=self.dense_block1(x)
        x1=self.trans_block1(x1_m)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        #print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)

        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)

        ## Classifier  ##

        x4_in = torch.cat([x3_2, x3], 1)
        x4=(self.dense_block4(x4_in))
        x4=self.trans_block4(x4)
        in_rs128 = torch.cat([x4, x2], 1)
        res_128 = self.res_est(in_rs128)
        conf_128 = self.conf_res(torch.cat([x4, res_128], 1))

        x5_in=torch.cat([x4, x2,res_128*conf_128], 1)

        x5=(self.dense_block5(x5_in))
        x5=self.trans_block5(x5)
        in_rs256 = torch.cat([x5, x1], 1)
        res_256 = self.res_est(in_rs256)
        conf_256 = self.conf_res(torch.cat([x5, res_256], 1))

        x6_in=torch.cat([x5, x1,res_256*conf_256], 1)

        x6=(self.dense_block6(x6_in))
        x6=(self.trans_block6(x6))
        x6=(self.dense_block6_1(x6))
        x6=(self.trans_block6_1(x6))
        
        
        x7=self.relu((self.conv_refin(x6)))
        residual=self.tanh(self.refine3(x7))
        x7_in_2 = torch.cat([x6,x6,residual],1)
        conf_512 = self.conf_res(x7_in_2)
        clean = x - residual
        clean1=self.relu(self.refineclean1(clean))
        clean2=self.tanh(self.refineclean2(clean1))
        clean128 = x_128 - res_128
        clean128 = self.relu(self.refineclean1(clean128))
        clean128 = self.tanh(self.refineclean2(clean128))
        clean256 = x_256 - res_256
        clean256 = self.relu(self.refineclean1(clean256))
        clean256 = self.tanh(self.refineclean2(clean256))

        return residual, clean2, clean128, clean256, conf_128, conf_256, conf_512
