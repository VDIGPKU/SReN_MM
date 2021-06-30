// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include "nms_poly_cuda_kernel.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>


#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
<< std::endl; cudaDeviceSynchronize(); } while (0)


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;
float const EPS = 1e-8;
const int maxisn = 10;

__device__ inline int dcmp(const float &x) {
    if(x > EPS) return 1;
    return x < -EPS ? -1 : 0;
}
template <typename T> __device__ void inline swap(T& a, T& b) {
	T c(a); a=b; b=c;
}

//template <typename T> __device__ void inline swap(T a, T b) {
//	    T c(a); a=b; b=c;
//}
__device__ inline float cross(const float & ax,const float & ay, \
   const float & bx,const float & by,const float & cx,const float & cy) {
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
}

__device__  inline void LineCross(const float & ax, const float & ay,const float & bx, \
  const float & by,const float & cx, const float & cy,const float & dx, \
  const float & dy, float& tx,float& ty) {
    float u = cross(ax,ay,bx,by,cx,cy), v = cross(bx,by,ax,ay,dx,dy);
    tx = (cx * v + dx * u) / (u + v);
    ty = (cy * v + dy * u) / (u + v);
}

__device__ inline float PolygonArea(float *px,float *py, int n) {
  float s = py[0] * (px[n - 1] - px[1]);
  // px[n] = px[0];
  // py[n] = py[0];
  for(int i = 1; i < n; ++i) {
    s += py[i] * (px[i - 1] - px[i + 1]);
  }
  return abs(s * 0.5);
}

__device__  inline float CPIA(float *ax, float *ay,float *bx,float *by) { //ConvexPolygonIntersectArea
  int na = 3, nb = 3;
  float px[maxisn],py[maxisn];
  float tmpx[maxisn],tmpy[maxisn];
  int i, j, tn, sflag, eflag;
  ax[na] = ax[0], bx[nb] = bx[0];
  ay[na] = ay[0], by[nb] = by[0];
 // memcpy(px,bx,sizeof(float)*(nb+1));
  for(int i = 0;i <= nb; ++i) {
      px[i] = bx[i];
      py[i] = by[i];
  }
 // memcpy(py,by,sizeof(float)*(nb+1));
  for(i = 0; i < na && nb > 2; ++i) {
    sflag = dcmp(cross(ax[i],ay[i],ax[i + 1],ay[i + 1], px[0],py[0]));
    for(j = tn = 0; j < nb; ++ j, sflag = eflag) {
      if(sflag >= 0) {
        tmpx[tn] = px[j];
        tmpy[tn++] = py[j];
      }
      eflag = dcmp(cross(ax[i], ay[i], ax[i + 1], ay[i + 1], px[j + 1], py[j + 1]));
      if((sflag ^ eflag) == -2) {
          LineCross(ax[i], ay[i], ax[i + 1], ay[i+1], px[j], py[j], px[j + 1], py[j+1], tmpx[tn],tmpy[tn]),tn++;
        }
    }
   // memcpy(px,tmpx,sizeof(float)*tn);
   // memcpy(py,tmpy,sizeof(float)*tn);
   for(int k=0;k<tn;k++) {
     px[k]=tmpx[k];
     py[k]=tmpy[k];
   }
    nb = tn, px[nb] = px[0],py[nb] = py[0];
  }
  //px[3] = px[0];
	//py[3] = py[0];
  return nb < 3? 0.0 : PolygonArea(px, py, nb);
}
__device__  inline float SPIA(float const * const ax, float const * const ay, float const * const bx, float const * const by) { //SimplePolygonIntersectArea
  float t1x[4],t1y[4], t2x[4],t2y[4];
  float res = 0, if_clock_t1, if_clock_t2;
  //a[na] = t1[0] = a[0], b[nb] = t2[0] = b[0];
  t1x[0] = ax[0];
  t1y[0] = ay[0];
  t2x[0] = bx[0];
  t2y[0] = by[0];
  for(int i = 2; i < 4; ++ i) {
    t1x[1] = ax[i - 1];
    t1y[1] = ay[i - 1];
    t1x[2] = ax[i];
    t1y[2] = ay[i];
    if_clock_t1 = dcmp(cross(t1x[0],t1y[0],t1x[1],t1y[1],t1x[2],t1y[2]));
    if(if_clock_t1 < 0) {
        swap(t1x[1], t1x[2]);
        swap(t1y[1], t1y[2]);
    }
    for(int j = 2; j < 4; ++j) {
        t2x[1] = bx[j - 1],t2y[1] = by[j - 1];
        t2x[2] = bx[j],t2y[2] = by[j];
        if_clock_t2 = dcmp(cross(t2x[0],t2y[0],t2x[1],t2y[1], t2x[2],t2y[2]));
        if(if_clock_t2 < 0) {
            swap(t2x[1], t2x[2]);
            swap(t2y[1], t2y[2]);
        }
        res += CPIA(t1x, t1y,t2x,t2y) * if_clock_t1 * if_clock_t2;
    }
  }
  return res;
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float ax[5], ay[5], bx[5], by[5];
  for (int i = 0; i < 4; ++i) {
    ax[i] = a[i * 2];
    ay[i] = a[i * 2 + 1];
    bx[i] = b[i * 2];
    by[i] = b[i * 2 + 1];
  }
	ax[4] = ax[0];
	ay[4] = ay[0];
	bx[4] = bx[0];
	by[4] = by[0];

	float interS =SPIA(ax, ay, bx, by);
  float Sa = PolygonArea(ax, ay, 4);
  float Sb = PolygonArea(bx, by, 4);
  // printf("a %.0f %.0f %.0f %.0f\nb %.0f %.0f %.0f %.0f\n %.3f %.3f %.3f %.3f\n\n", ax[0], ay[0], ax[2], ay[2],bx[0], by[0], bx[2], by[2], interS, Sa, Sb, interS / (Sa + Sb - interS));
  return interS / (Sa + Sb - interS);
}

__global__ void nms_poly_kernel(int n_boxes, float nms_overlap_thresh,
                                float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 9];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 9 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
    block_boxes[threadIdx.x * 9 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
    block_boxes[threadIdx.x * 9 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
    block_boxes[threadIdx.x * 9 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
    block_boxes[threadIdx.x * 9 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
    block_boxes[threadIdx.x * 9 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
    block_boxes[threadIdx.x * 9 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
    block_boxes[threadIdx.x * 9 + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    block_boxes[threadIdx.x * 9 + 8] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];

	}
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 9;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
			if (devIoU(cur_box, block_boxes + i * 9) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void nms_poly_cuda_compute(int* keep_out, int* num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh) {

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_poly_kernel<<<blocks, threads>>>(boxes_num,
                                       nms_overlap_thresh,
                                       boxes_dev,
                                       mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int* keep_out_cpu = new int[boxes_num];
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out_cpu[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  // copy keep_out_cpu to keep_out on gpu
  CUDA_WARN(cudaMemcpy(keep_out, keep_out_cpu, boxes_num * sizeof(int),cudaMemcpyHostToDevice));

  // *num_out = num_to_keep;

  CUDA_WARN(cudaMemcpy(num_out, &num_to_keep, 1 * sizeof(int),cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));

  // release cpu memory
  delete []keep_out_cpu;

}
