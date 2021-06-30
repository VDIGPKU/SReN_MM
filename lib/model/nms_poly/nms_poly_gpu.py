from __future__ import absolute_import
import torch
import numpy as np
from ._ext import nms_poly
import pdb


def nms_poly_gpu(dets, thresh):
    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()
    nms_poly.nms_poly_cuda(keep, dets, num_out, thresh)
    keep = keep[:num_out[0]]
    return keep
