from .xcodd_decoder import XcoddDecoder_Enhance_bottom
from .transnext import transnext_micro
import torch
import torch.nn as nn
import torch.nn.functional as F


class SG_Net(nn.Module):
    def __init__(self, n_class=1,drop_path_rate=0.1,img_size=256, deep_supervision=True):
        super(SG_Net, self).__init__()
        self.deep_supervision = deep_supervision
        self.drop_path_rate = drop_path_rate
        self.img_size = img_size
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = transnext_micro(drop_path_rate=drop_path_rate,img_size=img_size)  # [48, 96, 192, 384]
        path = r'/home/files/liutong/a0/xcodd/MSDUNet/models/pretrin_path/transnext_micro_224_1k.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = XcoddDecoder_Enhance_bottom(channels=[384, 192, 96, 48])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(384, n_class, 1)
        self.out_head2 = nn.Conv2d(192, n_class, 1)
        self.out_head3 = nn.Conv2d(96, n_class, 1)
        self.out_head4 = nn.Conv2d(48, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone.forward_intermediates(x)
        
        # decoder
        x4_o, x3_o, x2_o, x1_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x4_o)
        p2 = self.out_head2(x3_o)
        p3 = self.out_head3(x2_o)
        p4 = self.out_head4(x1_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        if self.deep_supervision:
            return p1, p2, p3, p4
        else:
            return p4