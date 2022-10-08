# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmdet.models import SELayer
from mmengine.model import BaseModule, Sequential
from torch import nn

from mmrotate.models.necks import RepConvModule
from mmrotate.models.utils import RepAlignConv, WeightedRepAlignConv


def test_repblock():
    # Test RepVGGBlock with in_channels != out_channels, stride = 1
    block = RepConvModule(5, 10, stride=1)
    block.eval()
    x = torch.randn(1, 5, 16, 16)
    x_out_not_deploy = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_1x1')
    assert hasattr(block, 'branch_3x3')
    assert hasattr(block, 'branch_norm')
    assert x_out_not_deploy.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with in_channels == out_channels, stride = 1
    block = RepConvModule(12, 12, stride=1)
    block.eval()
    x = torch.randn(1, 12, 8, 8)
    x_out_not_deploy = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert not hasattr(block, 'branch_reparam')
    assert x_out_not_deploy.shape == torch.Size((1, 12, 8, 8))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 12, 8, 8))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with in_channels == out_channels, stride = 2
    block = RepConvModule(16, 16, stride=2)
    block.eval()
    x = torch.randn(1, 16, 8, 8)
    x_out_not_deploy = block(x)
    assert block.branch_norm is None
    assert x_out_not_deploy.shape == torch.Size((1, 16, 4, 4))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 16, 4, 4))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with padding == dilation == 2
    block = RepConvModule(14, 14, stride=1, padding=2, dilation=2)
    block.eval()
    x = torch.randn(1, 14, 16, 16)
    x_out_not_deploy = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert x_out_not_deploy.shape == torch.Size((1, 14, 16, 16))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 14, 16, 16))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with groups = 2
    block = RepConvModule(4, 4, stride=1, groups=2)
    block.eval()
    x = torch.randn(1, 4, 5, 6)
    x_out_not_deploy = block(x)
    assert x_out_not_deploy.shape == torch.Size((1, 4, 5, 6))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 4, 5, 6))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with deploy == True
    block = RepConvModule(8, 8, stride=1, deploy=True)
    assert isinstance(block.branch_reparam, nn.Conv2d)
    assert not hasattr(block, 'branch_3x3')
    assert not hasattr(block, 'branch_1x1')
    assert not hasattr(block, 'branch_norm')
    x = torch.randn(1, 8, 16, 16)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 8, 16, 16))


def test_repalign():
    block = WeightedRepAlignConv(10, 3, strides=[1])
    block.init_weights()
    block.eval()
    x = torch.randn(1, 10, 16, 16)
    anchor = torch.randn(256, 5)
    scores = torch.randn(1, 1, 16, 16)
    x_out_not_deploy = block([x], [[anchor]], [scores])
    assert not hasattr(block, 'deform_conv_reparam')
    assert hasattr(block, 'deform_conv_1x1')
    assert hasattr(block, 'deform_conv_3x3')
    # assert hasattr(block, 'branch_norm')
    assert x_out_not_deploy[0].shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block([x], [[anchor]], [scores])
    assert x_out_deploy[0].shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(
        x_out_not_deploy[0], x_out_deploy[0], atol=1e-5, rtol=1e-4)

    # # Test RepVGGBlock with in_channels == out_channels, stride = 1
    # block = RepConvModule(12, 12, stride=1)
    # block.eval()
    # x = torch.randn(1, 12, 8, 8)
    # x_out_not_deploy = block(x)
    # assert isinstance(block.branch_norm, nn.BatchNorm2d)
    # assert not hasattr(block, 'branch_reparam')
    # assert x_out_not_deploy.shape == torch.Size((1, 12, 8, 8))
    # block.switch_to_deploy()
    # assert block.deploy is True
    # x_out_deploy = block(x)
    # assert x_out_deploy.shape == torch.Size((1, 12, 8, 8))
    # assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)
    #
    # # Test RepVGGBlock with in_channels == out_channels, stride = 2
    # block = RepConvModule(16, 16, stride=2)
    # block.eval()
    # x = torch.randn(1, 16, 8, 8)
    # x_out_not_deploy = block(x)
    # assert block.branch_norm is None
    # assert x_out_not_deploy.shape == torch.Size((1, 16, 4, 4))
    # block.switch_to_deploy()
    # assert block.deploy is True
    # x_out_deploy = block(x)
    # assert x_out_deploy.shape == torch.Size((1, 16, 4, 4))
    # assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)
    #
    # # Test RepVGGBlock with padding == dilation == 2
    # block = RepConvModule(14, 14, stride=1, padding=2, dilation=2)
    # block.eval()
    # x = torch.randn(1, 14, 16, 16)
    # x_out_not_deploy = block(x)
    # assert isinstance(block.branch_norm, nn.BatchNorm2d)
    # assert x_out_not_deploy.shape == torch.Size((1, 14, 16, 16))
    # block.switch_to_deploy()
    # assert block.deploy is True
    # x_out_deploy = block(x)
    # assert x_out_deploy.shape == torch.Size((1, 14, 16, 16))
    # assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)
    #
    # # Test RepVGGBlock with groups = 2
    # block = RepConvModule(4, 4, stride=1, groups=2)
    # block.eval()
    # x = torch.randn(1, 4, 5, 6)
    # x_out_not_deploy = block(x)
    # assert x_out_not_deploy.shape == torch.Size((1, 4, 5, 6))
    # block.switch_to_deploy()
    # assert block.deploy is True
    # x_out_deploy = block(x)
    # assert x_out_deploy.shape == torch.Size((1, 4, 5, 6))
    # assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)
    #
    # # Test RepVGGBlock with deploy == True
    # block = RepConvModule(8, 8, stride=1, deploy=True)
    # assert isinstance(block.branch_reparam, nn.Conv2d)
    # assert not hasattr(block, 'branch_3x3')
    # assert not hasattr(block, 'branch_1x1')
    # assert not hasattr(block, 'branch_norm')
    # x = torch.randn(1, 8, 16, 16)
    # x_out = block(x)
    # assert x_out.shape == torch.Size((1, 8, 16, 16))
