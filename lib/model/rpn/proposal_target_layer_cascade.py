from __future__ import absolute_import
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
import numpy as np
import numpy.random as npr
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch_reg
import pdb

class ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses

        self.BBOX_NORMALIZE_MEANS_REG = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS_REG)
        self.BBOX_NORMALIZE_STDS_REG = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS_REG)
        self.BBOX_INSIDE_WEIGHTS_REG = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS_REG)
    def forward(self, all_rois, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS_REG = self.BBOX_NORMALIZE_MEANS_REG.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS_REG = self.BBOX_NORMALIZE_STDS_REG.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS_REG = self.BBOX_INSIDE_WEIGHTS_REG.type_as(gt_boxes)

        # gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        # gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        # all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        if cfg.TRAIN.ENABLE_OHEM:
            rois_per_image = all_rois.shape[1]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
            fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
            fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, targets_sign,\
        poly_targets, poly_inside_weights, gt_sample = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        poly_outside_weights = (poly_inside_weights > 0).float()
        # return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights\
        #     , poly_targets, poly_inside_weights, poly_outside_weights
        return rois, labels, poly_targets, poly_inside_weights, poly_outside_weights, gt_sample, targets_sign

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, target_sign_data, poly_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch

        poly_targets = poly_target_data.new(batch_size, rois_per_image, 12).zero_()
        poly_inside_weights = poly_target_data.new(poly_targets.size()).zero_()
        target_sign = poly_target_data.new(batch_size, rois_per_image, 8).zero_()
        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                poly_targets[b, ind, :] = poly_target_data[b, ind, :]
                target_sign[b, ind, :] = target_sign_data[b, ind, :]
                poly_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS_REG

        return target_sign, poly_targets, poly_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        if cfg.QUAD_MODE:
            p = 12
        else:
            p = 4
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == p

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets_sign, poly_targets = bbox_transform_batch_reg(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            poly_targets = ((poly_targets - self.BBOX_NORMALIZE_MEANS_REG.expand_as(poly_targets))
                            / self.BBOX_NORMALIZE_STDS_REG.expand_as(poly_targets))

        return targets_sign, poly_targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        if cfg.QUAD_MODE:
            p = 12
        else:
            p = 4
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,-1].contiguous().view(-1).index(offset.view(-1))\
                                                            .view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, p + 1).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        targets_sign_data, poly_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:p])

        targets_sign, poly_targets, poly_inside_weights = \
                self._get_bbox_regression_labels_pytorch(targets_sign_data, poly_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, targets_sign, \
               poly_targets, poly_inside_weights, gt_rois_batch[:,:,:p]
