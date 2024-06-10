/*
 * Copyright (c) 2024, Alessio Medaglini and Biagio Peccerillo
 *
 * This file is part of GPU-MOT.
 *
 * GPU-MOT is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * GPU-MOT is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with GPU-MOT. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
inline void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

constexpr int maxLifeDuration = 5;
constexpr int maxNotDetection = 5;
constexpr int minDetection = 3;

void copyToConstant(const void* offsets, const void* rows, const void* cols, size_t size);

__global__ void associateAll(float* payload, int sizePayload, float* measures, int sizeMeasureElem, 
                                 const int numMeasures, int* associations, int* isAssoc);

__global__ void associateAllBIG(float* payloads, int sizePayload, float* measures, int sizeMeasureElem,
                                const int numMeasures, int* associations, int* isAssoc, float* curr_dist);
__global__ void associateAllBIG2(float* payloads, int sizePayload, float* measures, int sizeMeasureElem,
                                const int numMeasures, int* associations, int* isAssoc, float* distThreshold);
__global__ void calculateDistThreshold(float* payloads, int sizePayload, float* measures, int sizeMeasureElem,
                              const int numMeasures, int* associations, int* isAssoc, float* distThreshold);

                                
__device__ void matrixAddition(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols);
__device__ void matrixSubtraction(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols);
__device__ void matrixMultTranspose(const float A[6][6], const float T[6][6], float C[6][6], const int rowsA, const int colsA, const int rowT);
__device__ void matrixMultiplication(const float A[6][6], const float B[6][6], float C[6][6], const int rowsA, const int colsA, const int colsB);

// kernel launchers
void predictKFs_kernel(const int gridDim, const dim3 blockDim, float* payloads, int sizePayload);
void updateKFs_kernel(const int gridDim, const dim3 blockDim, float* payloads, int sizePayload, float* measures,
                      int sizeMeasureElem, int* associations, float* residuals);
void createTracks_kernel(const int gridDim, const dim3 blockDim, float* zeroPayload, float* payloads,
		         int sizePayload, float* measures, int sizeMeasureElem, const int numMeasures,
			 int* isAssoc);

template <int SIZE>
__device__ void matrixAddition(const float A[SIZE][SIZE], const float B[SIZE][SIZE], float C[SIZE][SIZE])
{
    C[threadIdx.y][threadIdx.x] = A[threadIdx.y][threadIdx.x] + B[threadIdx.y][threadIdx.x];
}
template <int SIZE>
__device__ void matrixSubtraction(const float A[SIZE][SIZE], const float B[SIZE][SIZE], float C[SIZE][SIZE])
{
    C[threadIdx.y][threadIdx.x] = A[threadIdx.y][threadIdx.x] - B[threadIdx.y][threadIdx.x];
}
template <int SIZE>
__device__ void matrixMultTransposeConst(const float A[SIZE][SIZE], const float T[SIZE][SIZE], float C[SIZE][SIZE])
{
    float cvalue = 0;
#pragma unroll
    for (int k = 0; k < SIZE; k++) 
        cvalue += A[threadIdx.y][k] * T[threadIdx.x][k];

    C[threadIdx.y][threadIdx.x] = cvalue;
}
template <int SIZE>
__device__ void matrixMultConst(const float A[SIZE][SIZE], const float B[SIZE][SIZE], float C[SIZE][SIZE])
{
    float cvalue = 0;
#pragma unroll
    for (int k = 0; k < SIZE; k++) 
        cvalue += A[threadIdx.y][k] * B[k][threadIdx.x];

    C[threadIdx.y][threadIdx.x] = cvalue;
}
template <int SIZE>
__device__ void matVecMultConst(const float A[SIZE][SIZE], const float x[SIZE], float y[SIZE])
{
	float cvalue = 0;
	if(threadIdx.y == 0)
	{
#pragma unroll
		for (int k = 0; k < SIZE; ++k)
			cvalue += A[threadIdx.x][k] * x[k];

		y[threadIdx.x] = cvalue;
	}
	__syncthreads();
}
template <int SIZE>
__device__ void vectorAdditionConst(const float x[SIZE], const float y[SIZE], float z[SIZE])
{
	if(threadIdx.y == 0)
		z[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
	__syncthreads();
}
template <int SIZE>
__device__ void vectorSubtractionConst(const float x[SIZE], const float y[SIZE], float z[SIZE])
{
	if(threadIdx.y == 0)
		z[threadIdx.x] = x[threadIdx.x] - y[threadIdx.x];
	__syncthreads();
}

constexpr int F_index = 0;
constexpr int H_index = 1;
constexpr int P_index = 2;
constexpr int G_index = 3;
constexpr int Q_index = 4;
constexpr int R_index = 5;

constexpr int first_index = 6;
constexpr int life_time_index = 7;
constexpr int track_state_index = 8;
constexpr int serial_miss_index = 9;
constexpr int attempt_time_index = 10;

constexpr int K_index = 11;
constexpr int S_index = 12;
constexpr int Sinv_index = 13;
constexpr int P_predict_index = 14;
constexpr int x_filter_index = 15;
constexpr int x_predict_index = 16;
constexpr int z_predict_index = 17;
constexpr int z_measured_index = 18;
constexpr int bb_size_index = 19;

constexpr int dev_offset(int index)
{

    return (index == F_index) ? 0 :
           (index == H_index) ? 36 :
           (index == P_index) ? 72 :
           (index == G_index) ? 108 :
           (index == Q_index) ? 126 :
           (index == R_index) ? 135 :
           (index == first_index) ? 171 :
           (index == life_time_index) ? 172 :
           (index == track_state_index) ? 173 :
           (index == serial_miss_index) ? 174 :
           (index == attempt_time_index) ? 175 :
           (index == K_index) ? 176 :
           (index == S_index) ? 212 :
           (index == Sinv_index) ? 248 :
           (index == P_predict_index) ? 284 :
           (index == x_filter_index) ? 320 :
           (index == x_predict_index) ? 326 :
           (index == z_predict_index) ? 332 :
           (index == z_measured_index) ? 338 :
           (index == bb_size_index) ? 344 : -1;
}

#endif // _CUDA_KERNEL_H_
