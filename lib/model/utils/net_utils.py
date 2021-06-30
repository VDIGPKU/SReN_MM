import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from lib.model.utils.config import cfg
from lib.model.roi_crop.functions.roi_crop import RoICropFunction
from lib.model.rpn.bbox_transform import bbox_transform_inv_reg
import cv2
import pdb
import random


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def draw_polygon(ori_img, pts, is_copy=True):
    if is_copy:
        img = ori_img.copy()
    else:
        img = ori_img
    if type(pts) != np.ndarray or len(pts) == 0:
        return img
    pts = pts.reshape((-1, 1, 2))
    # print('pts', pts)
    cv2.polylines(img, [pts], True, (255, 0, 0), thickness=8)
    return img


def vis_detections(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        quad = dets[i, :8]
        quad = quad.astype(int)
        # score = dets[i, -1]
        draw_polygon(im, quad, is_copy=False)
        score = dets[i, -1]
        cv2.putText(im, '%.3f' % (score), (quad[0], quad[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 0, 255), thickness=2)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def adjust_learning_rate_warmup(optimizer, lr, epoch, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch == 1 and iteration < epoch_size/2:
        lr = 1e-5 + (lr - 1e-5) * iteration * 2 / (epoch_size)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def save_checkpoint(state, filename):
    torch.save(state, filename)


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1],
                    reduce=True):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    if type(abs_in_box_diff) == torch.cuda.FloatTensor or type(abs_in_box_diff) == torch.Tensor:
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).float()
    else:
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    if reduce:
        loss_box = loss_box.mean()
    return loss_box

def _smooth_ln_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    in_loss_box = (abs_in_box_diff + 1) * torch.log(abs_in_box_diff + 1) - abs_in_box_diff
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def edge_target_layer(rois, poly_deltas, gt_boxes, labels):
    # print(rois.shape, poly_deltas.shape, gt_boxes.shape, labels.shape, '##########')
    pred_poly = bbox_transform_inv_reg(rois, poly_deltas, 1)
    gt_poly = gt_boxes[:, 4:12]
    #  cos_sum = cos_sum.astype(np.float32)
    tmp_sum = []
    for j in range(0, 8, 2):
        x1 = pred_poly[:, (j + 2) % 8] - pred_poly[:, j]
        y1 = pred_poly[:, (j + 3) % 8] - pred_poly[:, j + 1]

        x2 = gt_poly[:, (j + 2) % 8] - gt_poly[:, j]
        y2 = gt_poly[:, (j + 3) % 8] - gt_poly[:, j + 1]
        cos_theta = (x1 * x2 + y1 * y2) / (torch.sqrt(x1 * x1 + y1 * y1) * torch.sqrt(x2 * x2 + y2 * y2) + 1e-8)
        tmp_sum.append((1 - cos_theta).view(-1, 1))
    cos_sum = torch.mean(torch.cat(tmp_sum, dim=1), dim=1)
    labels = (labels > 0).float()
    edge_loss = labels * cos_sum
    # assert(not np.isnan(edge_loss.mean().data.cpu().numpy()[0])) 
    return edge_loss.mean()


def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
        pre_pool_size = cfg.POOLING_SIZE * 2
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)
        crops = F.max_pool2d(crops, 2, 2)
    else:
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)

    return crops, grid


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def _affine_theta(rois, input_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([ \
        (y2 - y1) / (height - 1),
        zero,
        (y1 + y2 - height + 1) / (height - 1),
        zero,
        (x2 - x1) / (width - 1),
        (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta


def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2  # random.randint(1, 8)
    H = 5  # random.randint(1, 8)
    W = 4  # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()

    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]

    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:, :, :, 1], grid_clone.data[:, :, :, 0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()
