/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

namespace wisp {
typedef unsigned int uint;

__device__ int32_t
hash_index(
    const int3 pos,
    const int32_t resolution,
    const int32_t codebook_size
){
    int32_t index = 0;

    constexpr uint32_t primes[3] = { 1u, 2654435761u, 805459861u };

    if (resolution < codebook_size &&
        resolution * resolution < codebook_size &&
        resolution * resolution * resolution < codebook_size) {
        index = pos.x +
                pos.y * resolution +
                pos.z * resolution * resolution;
    } else {
        index = (pos.x * primes[0] ^
                 pos.y * primes[1] ^
                 pos.z * primes[2]) % codebook_size;
    }
    return index;
}

__device__ float
clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__global__ void
hashgrid_interpolate_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* coords,
    const float* codebook,
    float* feats
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    for (int64_t i=tidx; i<num_coords; i+=stride) {

        float3 x = make_float3(clamp(resolution * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5),
                               clamp(resolution * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5),
                               clamp(resolution * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        float3 x_ = make_float3(x.x - (float) pos.x, x.y - (float) pos.y, x.z - (float) pos.z);
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float c000 = _x.x * _x.y * _x.z;
        float c001 = _x.x * _x.y * x_.z;
        float c010 = _x.x * x_.y * _x.z;
        float c011 = _x.x * x_.y * x_.z;
        float c100 = x_.x * _x.y * _x.z;
        float c101 = x_.x * _x.y * x_.z;
        float c110 = x_.x * x_.y * _x.z;
        float c111 = x_.x * x_.y * x_.z;

        int32_t corner_idx[8];
#       pragma unroll
        for (int j=0; j<8; ++j) {
            int3 corner;
            corner.x = pos.x + ((j & 4) >> 2);
            corner.y = pos.y + ((j & 2) >> 1);
            corner.z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index(corner, resolution, codebook_size);
        }

        for (uint64_t j=0; j<feature_dim; ++j) {
            float c[8];
            for (int m = 0; m < 8; m++)
            {
                c[m] = codebook[corner_idx[m] * feature_dim + j];
            }

            float feat =
                c[0] * c000 +
                c[1] * c001 +
                c[2] * c010 +
                c[3] * c011 +
                c[4] * c100 +
                c[5] * c101 +
                c[6] * c110 +
                c[7] * c111;
            feats[num_lods*i*feature_dim+feature_dim*lod_idx+j] = feat;
        }
    }
}

void hashgrid_interpolate_cuda_impl(
    int64_t num_coords,
    int32_t codebook_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor feats){

    int num_threads = 512;

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
    auto stream = at::cuda::getCurrentCUDAStream();
    hashgrid_interpolate_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
        num_coords,
        codebook_size,
        feature_dim,
        resolution,
        lod_idx,
        num_lods,
        coords.data_ptr<float>(),
        codebook.data_ptr<float>(),
        feats.data_ptr<float>()
    );
}
// --- Yasasa - start - interp gradients
__device__ void
calc_grad_wrt_x_(
    int64_t feature_index,
    int64_t feature_dim,
    float3 x_, float3 _x,
    float resolution,
    int32_t corner_idx[8],
    float gradout,
    float* codebook,
    float* grad_x_){

    float c[8];

    #pragma unroll
    for(int i = 0; i < 8; i++){
        c[i] = codebook[corner_idx[i]*feature_dim + feature_index];
    }

    gradout *= resolution / 2.f;

    auto x_grad = _x.y * _x.z * (c[4] - c[0])
                + x_.y * _x.z * (c[6] - c[2])
                + _x.y * x_.z * (c[5] - c[1])
                + x_.y * x_.z * (c[7] - c[3]);

    atomicAdd(grad_x_, gradout * x_grad);

    auto y_grad = _x.x * _x.z * (c[2] - c[0]) +
                  _x.x * x_.z * (c[3] - c[1]) +
                  x_.x * _x.z * (c[6] - c[4]) +
                  x_.x * x_.z * (c[7] - c[5]);

    atomicAdd(grad_x_ + 1, gradout * y_grad);


    auto z_grad = _x.x * _x.y * (c[1] - c[0]) +
                  _x.x * x_.y * (c[3] - c[2]) +
                  x_.x * _x.y * (c[5] - c[4]) +
                  x_.x * x_.y * (c[7] - c[6]);

    atomicAdd(grad_x_ + 2, gradout * z_grad);
}
// gradient of interpolated features w.r.t grid points
// --- Yasasa - end - interp gradients
__global__ void
hashgrid_interpolate_backward_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* coords,
    const float* grad_output, // N, feature_dim*num_lods
    float* grad_codebook, // codebook_size, feature_dim
    float* codebook,
    float* grad_coords // num_coords * num_features, 3
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    for (int64_t i=tidx; i<num_coords; i+=stride) {

        float3 x = make_float3(clamp(resolution * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5),
                               clamp(resolution * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5),
                               clamp(resolution * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        float3 x_ = make_float3(x.x - (float) pos.x, x.y - (float) pos.y, x.z - (float) pos.z);
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float coeffs[8];
        coeffs[0] = _x.x * _x.y * _x.z;
        coeffs[1] = _x.x * _x.y * x_.z;
        coeffs[2] = _x.x * x_.y * _x.z;
        coeffs[3] = _x.x * x_.y * x_.z;
        coeffs[4] = x_.x * _x.y * _x.z;
        coeffs[5] = x_.x * _x.y * x_.z;
        coeffs[6] = x_.x * x_.y * _x.z;
        coeffs[7] = x_.x * x_.y * x_.z;

        int32_t corner_idx[8];
#       pragma unroll
        for (int j=0; j<8; ++j) {
            int3 corner;
            corner.x = pos.x + ((j & 4) >> 2);
            corner.y = pos.y + ((j & 2) >> 1);
            corner.z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index(corner, resolution, codebook_size);
        }


        for (uint64_t j=0; j<feature_dim; ++j) {
            float gradout =  grad_output[i*num_lods*feature_dim + lod_idx*feature_dim + j];
            calc_grad_wrt_x_(j, feature_dim, x_, _x, resolution, corner_idx, gradout, codebook, grad_coords + i*3);
#           pragma unroll
            for (int k=0; k<8; ++k) {
                float grad = gradout * coeffs[k];
                atomicAdd(grad_codebook + (corner_idx[k]*feature_dim + j), grad);
            }
        }
    }
}

void hashgrid_interpolate_backward_cuda_impl(
    int64_t num_coords,
    int32_t codebook_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor grad_codebook,
    at::Tensor codebook,
    at::Tensor grad_coords){

    int num_threads = 512;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_codebook));
    auto stream = at::cuda::getCurrentCUDAStream();
    hashgrid_interpolate_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
        num_coords,
        codebook_size,
        feature_dim,
        resolution,
        lod_idx,
        num_lods,
        coords.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_codebook.data_ptr<float>(),
        codebook.data_ptr<float>(),
        grad_coords.data_ptr<float>()
    );
}

} // namespace wisp
