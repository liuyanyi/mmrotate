import numpy as np
import torch
from matplotlib import pyplot as plt
from mmdet.models import L1Loss
from mmrotate.core.bbox.coder import CSLCoder

from mmrotate.models.losses import SmoothFocalLoss
import torch.nn.functional as F


def show_loss_3d(pred, target, loss):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-135)
    ax.plot_surface(pred.numpy() * 180 / np.pi, target.numpy() * 180 / np.pi, loss.numpy(), cmap='rainbow')
    # 坐标轴标题
    ax.set_xlabel('pred')
    ax.set_ylabel('target')
    ax.set_zlabel('loss')
    # plt.imshow(loss.detach().numpy())
    plt.show()


def loss_l1(pred, target):
    return torch.abs(pred - target)


def focal_loss(pred,
               target,
               gamma=2.0,
               alpha=0.25
               ):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    return loss


angle_coder = CSLCoder('le90', omega=1, radius=8)

target = np.arange(-90, 90, 1) * np.pi / 180.0
pred = np.arange(-90, 90, 1) * np.pi / 180.0

target = torch.from_numpy(target).unsqueeze(0).repeat((180, 1)).float()
pred = torch.from_numpy(pred).unsqueeze(1).repeat(1, 180).float()

flatten_target = target.view(-1)
flatten_pred = pred.view(-1)

l1 = loss_l1(flatten_pred, flatten_target)
l1 = l1.view(180, 180)/l1.max()
show_loss_3d(pred, target, l1)

csl_target = angle_coder.encode(flatten_target.unsqueeze(1))
csl_pred = angle_coder.encode(flatten_pred.unsqueeze(1))
loss_csl = focal_loss(csl_pred, csl_target)
loss_csl = loss_csl.sum(dim=1)
loss_csl = loss_csl.view(180, 180)/loss_csl.max()
loss_csl = loss_csl.sqrt()
show_loss_3d(pred, target, loss_csl)
