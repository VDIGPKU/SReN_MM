# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from lib.model.utils.config import cfg
from lib.model.nms_poly.nms_poly_gpu import nms_poly_gpu

def nms_poly(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    # ---numpy version---
    # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    # ---pytorch version---
    return nms_poly_gpu(dets, thresh)
