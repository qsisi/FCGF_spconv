import torch
from torch import nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SparseSequential(
                    conv3x3(inplanes, planes, stride, indice_key=indice_key),
                    nn.BatchNorm1d(planes, momentum=0.01, eps=1e-3),
                    nn.ReLU()
        )
        self.conv2 = spconv.SparseSequential(
                    conv3x3(planes, planes, indice_key=indice_key),
                    nn.BatchNorm1d(planes, momentum=0.01, eps=1e-3)
        )
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out

class FCGF_spconv(nn.Module):
    def __init__(self):
        super(FCGF_spconv, self).__init__()
        self.relu = nn.ReLU()
        self.skip_x = []
        ##### encoder #####
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(1, 32, kernel_size=5, stride=1, dilation=1, bias=False, indice_key="subm0"),
        )
        self.norm1 = nn.BatchNorm1d(32, momentum=0.01, eps=1e-3)
        self.block1 = SparseBasicBlock(inplanes=32, planes=32, indice_key="resnet1")

        self.conv2 = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, dilation=1, bias=False, indice_key="down1"),
        )
        self.norm2 = nn.BatchNorm1d(64, momentum=0.01, eps=1e-3)
        self.block2 = SparseBasicBlock(inplanes=64, planes=64, indice_key="resnet2")

        self.conv3 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, dilation=1, bias=False, indice_key="down2"),
        )
        self.norm3 = nn.BatchNorm1d(128, momentum=0.01, eps=1e-3)
        self.block3 = SparseBasicBlock(inplanes=128, planes=128, indice_key="resnet3")

        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(128, 256, kernel_size=3, stride=2, dilation=1, bias=False, indice_key="down3"),
        )
        self.norm4 = nn.BatchNorm1d(256, momentum=0.01, eps=1e-3)
        self.block4 = SparseBasicBlock(inplanes=256, planes=256, indice_key="resnet4")

        ##### decoder #####
        #### does the following InverseConv3d corresponds to MinkowskiConvolutionTranspose(in=256, out=128,
        # kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1]) ?
        self.conv4_tr = spconv.SparseSequential(
            spconv.SparseInverseConv3d(256, 128, kernel_size=3, bias=False, indice_key="down3"),
        )
        self.norm4_tr = nn.BatchNorm1d(128, momentum=0.01, eps=1e-3)
        self.block4_tr = SparseBasicBlock(inplanes=128, planes=128, indice_key="resnet3")

        self.conv3_tr = spconv.SparseSequential(
            spconv.SparseInverseConv3d(256, 64, kernel_size=3, bias=False, indice_key="down2"),
        )
        self.norm3_tr = nn.BatchNorm1d(64, momentum=0.01, eps=1e-3)
        self.block3_tr = SparseBasicBlock(inplanes=64, planes=64, indice_key="resnet2")

        self.conv2_tr = spconv.SparseSequential(
            spconv.SparseInverseConv3d(128, 64, kernel_size=3, bias=False, indice_key="down1"),
        )
        self.norm2_tr = nn.BatchNorm1d(64, momentum=0.01, eps=1e-3)
        self.block2_tr = SparseBasicBlock(inplanes=64, planes=64, indice_key="resnet1")

        self.conv1_tr = spconv.SparseSequential(
            spconv.SubMConv3d(96, 64, kernel_size=1, stride=1, dilation=1, bias=False, indice_key="subm0"),
        )

        self.final = spconv.SparseSequential(
            spconv.SubMConv3d(64, 32, kernel_size=1, stride=1, dilation=1, bias=False, indice_key="subm0"),
        )

    def forward(self, sp_tensor):

        out = self.conv1(sp_tensor)
        out = out.replace_feature(self.norm1(out.features))
        out = self.block1(out)
        self.skip_x.append(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv1 spatial shape -> {out.spatial_shape}')

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = self.block2(out)
        self.skip_x.append(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv2 spatial shape -> {out.spatial_shape}')

        out = self.conv3(out)
        out = out.replace_feature(self.norm3(out.features))
        out = self.block3(out)
        self.skip_x.append(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv3 spatial shape -> {out.spatial_shape}')

        out = self.conv4(out)
        out = out.replace_feature(self.norm4(out.features))
        out = self.block4(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv4 spatial shape -> {out.spatial_shape}')

        out = self.conv4_tr(out)
        out = out.replace_feature(self.norm4_tr(out.features))
        out = self.block4_tr(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv4_tr spatial shape -> {out.spatial_shape}')

        curr_cat_features = self.skip_x.pop().features
        out = out.replace_feature(torch.cat((out.features, curr_cat_features), dim=-1))

        out = self.conv3_tr(out)
        out = out.replace_feature(self.norm3_tr(out.features))
        out = self.block3_tr(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv3_tr spatial shape -> {out.spatial_shape}')

        curr_cat_features = self.skip_x.pop().features
        out = out.replace_feature(torch.cat((out.features, curr_cat_features), dim=-1))

        out = self.conv2_tr(out)
        out = out.replace_feature(self.norm2_tr(out.features))
        out = self.block2_tr(out)
        out = out.replace_feature(self.relu(out.features))
        # print(f'after conv2_tr spatial shape -> {out.spatial_shape}')

        curr_cat_features = self.skip_x.pop().features
        out = out.replace_feature(torch.cat((out.features, curr_cat_features), dim=-1))

        out = self.conv1_tr(out)
        out = out.replace_feature(self.relu(out.features))
        out = self.final(out)
        # print(f'after final spatial shape -> {out.spatial_shape}')

        ### normalize feature output ###
        out = out.replace_feature(F.normalize(out.features, p=2))

        return out

