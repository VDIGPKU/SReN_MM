from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
from lib.model.utils.config import cfg
from lib.model.rpn.generate_anchors import generate_anchors
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from lib.model.utils.net_utils import _smooth_l1_loss
import pdb

DEBUG = False

class BoxAnnotatorOHEM(nn.Module):
    """
    Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
    """
    def __init__(self, num_classes, num_reg_classes, roi_per_img):
        super(BoxAnnotatorOHEM, self).__init__()
        self._num_classes = num_classes
        self._num_reg_classes = num_reg_classes
        self._roi_per_img = roi_per_img

    def forward(self, cls_score, bbox_pred, labels, bbox_targets, inside_ws, outside_ws):

        per_roi_loss_cls = F.softmax(cls_score, dim=1).data
        
        # per_roi_loss_cls = per_roi_loss_cls[torch.arange(per_roi_loss_cls.shape[0]).long()]
        per_roi_loss_cls = per_roi_loss_cls[torch.arange(0,per_roi_loss_cls.shape[0]).cuda().long(), labels.long()]
        per_roi_loss_cls = -1 * torch.log(per_roi_loss_cls)
        per_roi_loss_cls = per_roi_loss_cls.view((-1,))

        per_roi_loss_bbox = _smooth_l1_loss(bbox_pred, bbox_targets, inside_ws, outside_ws, reduce=False)

        _, top_k_per_roi_loss = torch.sort(per_roi_loss_cls + per_roi_loss_bbox, descending=True)
        labels_ohem = labels
        labels_ohem[top_k_per_roi_loss[self._roi_per_img:]] = -1
        bbox_weights_inside_ohem = inside_ws
        bbox_weights_inside_ohem[top_k_per_roi_loss[self._roi_per_img:]] = 0

        bbox_weights_outside_ohem = outside_ws
        bbox_weights_outside_ohem[top_k_per_roi_loss[self._roi_per_img:]] = 0
        # print(type(bbox_targets), type(inside_ws), type(bbox_weights_inside_ohem), bbox_weights_inside_ohem == inside_ws, type(cls_score), type(bbox_pred)) 
        return labels_ohem, bbox_weights_inside_ohem, bbox_weights_outside_ohem

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
