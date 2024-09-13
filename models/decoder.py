import torch
import torch.nn as nn
from .modules import ASFF,ASPP,CBAMBlock,SEAttention,CARAFE,AIFI,AIFISP,AIFISPSE
from .kancbam import KANConv2dLayerConv,KANConv2dLayerConvCBAM,KANConv2dLayerConvCBAMSE
from .common import RepNCSPELAN4,RepNCSPELAN4CBAM,RepNCSPELAN4CBAM_1
class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]

class DecoderChannels(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(DecoderChannels, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, C5_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(C5_size, C4_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, C4_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, C3_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(C3_size, C3_size, kernel_size=3, stride=1, padding=1)

        self.P4_out=nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]
class DecoderCARAFE(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(DecoderCARAFE, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = CARAFE(feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]


class DecoderP3(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(DecoderP3, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_2_up=nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=2),
                                   nn.SiLU())

        self.P4_combine=nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P3_x_up=self.P3_2_up(P3_x)
        P4_x=self.P4_combine(P4_x+P3_x_up)

        return [P3_x, P4_x, P5_x]

class DecoderBiFPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(DecoderBiFPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_2_up=nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=2),
                                   nn.SiLU())

        self.P4_combine=nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P3_x_up=self.P3_2_up(P3_x)
        P4_x=self.P4_combine(P4_x+P3_x_up)

        return [P3_x, P4_x, P5_x]


class DecoderCBAM(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(DecoderCBAM, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.CBAM=CBAMBlock(channel=feature_size)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        P4_x=self.CBAM(P4_x)
        return [P3_x, P4_x, P5_x]


class KANDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(KANDecoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = KANConv2dLayerConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = KANConv2dLayerConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = KANConv2dLayerConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        self.ASPP_3=ASPP(256,[6,12,14])
        self.ASFF_1=ASFF(1,dim=[256,256,256])
        self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
                                    # nn.BatchNorm2d(feature_size),
                                    nn.ReLU())
        self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
                                  # nn.BatchNorm2d(feature_size),
                                  nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        CBAM_in=torch.cat((C3,P3_x),dim=1)
        # P3_x_temp=self.ASPP_3(P3_x)
        # P3_x= P3_x+P3_x_temp
        P3_x=self.P4_out(self.CBAM(CBAM_in))
        P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]


class ELENDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENDecoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]

class ELENCBAMDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENCBAMDecoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]

class ELENCBAMDecoder_1(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENCBAMDecoder_1, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]

class ELENCBAMDecoder_2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENCBAMDecoder_2, self).__init__()
        # self.AIFI = AIFI(c1=C5_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        # self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = RepNCSPELAN4CBAM_1(C5_size,feature_size,feature_size//2,feature_size//4)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]

class ELENCBAMASFFDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENCBAMASFFDecoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_1 = RepNCSPELAN4CBAM_1(C5_size,feature_size,feature_size//2,feature_size//4)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_2 = RepNCSPELAN4CBAM_1(feature_size, feature_size, feature_size // 2, feature_size // 4)
        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
                                  # nn.BatchNorm2d(feature_size),
                                  nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        P4_x=self.P4_out(P4_x)
        return [P3_x, P4_x, P5_x]


class ELENCBAMDecoder_real_2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(ELENCBAMDecoder_real_2, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = RepNCSPELAN4CBAM_1(C5_size,feature_size,feature_size//2,feature_size//4)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)

        P5_x = self.P5_2(P5_x)
        P5_upsampled_x = self.P5_upsampled(P5_x)


        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [C3, P4_x, P5_x]


class BiFPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(BiFPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2=RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_2 = RepNCSPELAN4CBAM_1(feature_size, feature_size, feature_size // 2, feature_size // 4)


        self.P3_2_up=nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=2),
                                   nn.SiLU())
        self.P4_3_block=RepNCSPELAN4CBAM_1(feature_size*3, feature_size, feature_size // 2, feature_size // 4)
        self.P4_out=nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        #
        # self.P4_combine=nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x_temp=P4_x
        P4_x = P5_upsampled_x + P4_x
        P4_x = self.P4_2(P4_x)
        # P4_x_temp_1=P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P3_x_up=self.P3_2_up(P3_x)

        P4_x=self.P4_out(self.P4_3_block(torch.cat((P4_x_temp,P4_x,P3_x_up),dim=1)))

        # P4_x=self.P4_combine(P4_x+P3_x_up)

        return [P3_x, P4_x, P5_x]

class AIFIDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFIDecoder, self).__init__()
        self.AIFI = AIFI(c1=C5_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs
        C5=self.AIFI(C5)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [C3, P4_x, P5_x]

class AIFISPDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFISPDecoder, self).__init__()
        self.AIFI = AIFISP(C5_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs
        C5 = self.AIFI(C5)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [C3, P4_x, P5_x]

class AIFISPSEDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFISPSEDecoder, self).__init__()
        self.AIFI = AIFISPSE(C5_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.CBAM=CBAMBlock(channel=C3_size+feature_size,kernel_size=3)
        # self.CBAM=KANConv2dLayerConvCBAMSE(C3_size+feature_size,C3_size+feature_size,kernel_size=3,padding=1)
        # self.ASPP_3=ASPP(256,[6,12,14])
        # self.ASFF_1=ASFF(1,dim=[256,256,256])
        # self.P3_out = nn.Sequential(nn.Conv2d(C3_size+feature_size, feature_size, kernel_size=1),
        #                             # nn.BatchNorm2d(feature_size),
        #                             nn.ReLU())
        # self.P4_out=nn.Sequential(nn.Conv2d(feature_size*2,feature_size, kernel_size=1),
        #                           # nn.BatchNorm2d(feature_size),
        #                           nn.ReLU())
    def forward(self, inputs):
        C3, C4, C5 = inputs
        C5 = self.AIFI(C5)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)
        # CBAM_in=torch.cat((C3,P3_x),dim=1)
        # # P3_x_temp=self.ASPP_3(P3_x)
        # # P3_x= P3_x+P3_x_temp
        # P3_x=self.P4_out(self.CBAM(CBAM_in))
        # P4_x=self.ASFF_1(P5_x,P4_x,P3_x)
        # P4_x=self.P4_out(P4_x)
        return [C3, P4_x, P5_x]

class AIFIAPDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFIAPDecoder, self).__init__()
        self.AIFI = AIFI(c1=feature_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        C5 = self.AIFI(P5_x)
        P5_upsampled_x = self.P5_upsampled(P5_x+C5)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_x = self.P4_2(P4_x)


        return [C3, P4_x, P5_x]

class AIFIMPDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFIMPDecoder, self).__init__()
        self.AIFI = AIFI(c1=feature_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_combine=nn.Conv2d(feature_size*2, feature_size, kernel_size=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        C5 = self.AIFI(P5_x)
        P5_upsampled_x = self.P5_upsampled(self.P5_combine(torch.cat((P5_x,C5),dim=1)))

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_x = self.P4_2(P4_x)


        return [C3, P4_x, P5_x]

class AIFISEAPDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFISEAPDecoder, self).__init__()
        self.AIFI = AIFISPSE(c1=feature_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        C5 = self.AIFI(P5_x)
        P5_upsampled_x = self.P5_upsampled(P5_x+C5)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_x = self.P4_2(P4_x)


        return [C3, P4_x, P5_x]

class AIFISEMPDecoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(AIFISEMPDecoder, self).__init__()
        self.AIFI = AIFISPSE(c1=feature_size, dropout=0.1)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_combine=nn.Conv2d(feature_size*2, feature_size, kernel_size=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = RepNCSPELAN4CBAM_1(feature_size,feature_size,feature_size//2,feature_size//4)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        C5 = self.AIFI(P5_x)
        P5_upsampled_x = self.P5_upsampled(self.P5_combine(torch.cat((P5_x,C5),dim=1)))

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_x = self.P4_2(P4_x)


        return [C3, P4_x, P5_x]