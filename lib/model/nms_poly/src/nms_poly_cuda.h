//void nms_poly_cuda(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
//          int boxes_dim, float nms_overlap_thresh, int device_id);

//int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
//             THCudaIntTensor *num_out, float nms_overlap_thresh);

void nms_poly_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
             THCudaIntTensor *num_out, float nms_overlap_thresh);