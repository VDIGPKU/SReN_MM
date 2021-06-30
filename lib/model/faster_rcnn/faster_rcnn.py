import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import ProposalTargetLayer
from lib.model.rpn.box_annotator_ohem import BoxAnnotatorOHEM
from lib.model.utils.net_utils import edge_target_layer
import time
import pdb
from lib.model.utils.net_utils import _smooth_l1_loss, _smooth_ln_loss, \
    _crop_pool_layer, _affine_grid_gen, _affine_theta

class FasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = RPN(self.dout_base_model)
        self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        num_reg_classes = (2 if class_agnostic else self.n_classes)
        self.box_annotator_ohem = BoxAnnotatorOHEM(self.n_classes, num_reg_classes, cfg.TRAIN.BATCH_ROIS_OHEM)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, gt_sample, target_sign = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            gt_sample = Variable(gt_sample.contiguous().view(-1, gt_sample.size(2)))
            rois_sign = Variable(target_sign.contiguous().view(-1).long())
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            gt_sample = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois_sign = None

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        if cfg.QUAD_MODE:
            p = 12
        else:
            p = 4

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / p), p)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, p))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        if self.n_classes > 2:
            cls_prob = F.softmax(cls_score, dim=1)
        else:
            cls_prob = F.sigmoid(cls_score)

        bbox_sign_score = self.RCNN_bbox_sign(pooled_feat)
        bbox_sign_score = bbox_sign_score.view(-1, 2)
        bbox_sign_prob = F.sigmoid(bbox_sign_score)
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_edge = 0
        RCNN_loss_sign = 0
        if self.training:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, inside_ws_ohem, outside_ws_ohem = self.box_annotator_ohem(
                    cls_score, bbox_pred.data, rois_label.data.clone(), rois_target.data, rois_inside_ws.data.clone(), rois_outside_ws.data.clone())
                labels_ohem = Variable(labels_ohem.view(-1).long())
                inside_ws_ohem = Variable(inside_ws_ohem)
                outside_ws_ohem = Variable(outside_ws_ohem)
                RCNN_loss_cls = F.cross_entropy(cls_score, labels_ohem, ignore_index=-1)
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, inside_ws_ohem, outside_ws_ohem, reduce=False)
                RCNN_loss_bbox = RCNN_loss_bbox.sum() / cfg.TRAIN.BATCH_ROIS_OHEM
                rois_label = labels_ohem
            else:
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
                # print((type(cls_score), type(rois_label)))
                # print((type(bbox_sign_score), type(rois_sign)))
                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, sigma=2)
                RCNN_loss_sign = F.cross_entropy(bbox_sign_score, rois_sign)
            if cfg.TRAIN.EDGE_LOSS:
                RCNN_loss_edge = edge_target_layer(rois.view(-1, 5), bbox_pred, gt_sample, rois_label)
            else:
                RCNN_loss_edge = RCNN_loss_cls
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        bbox_sign_prob = bbox_sign_prob.view(batch_size, -1, 2)

        # print(type(cls_prob), cls_prob.shape, type(bbox_sign_prob), bbox_sign_prob.shape)
        # print(type(RCNN_loss_bbox), type(RCNN_loss_sign))
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_sign, bbox_sign_prob, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
