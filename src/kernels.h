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

#include <cuda_runtime.h>
#include <cassert>

__global__ void predictKFs(float* payloads, int sizePayload);
__device__ void predictKF(float* payloads);
__global__ void updateKFs(float* payloads, int sizePayload, float* measures, int sizeMeasureElem, int* associations);
__device__ void updateKF(float* payload, float* measure);

__global__ void associateAll(float* payload, int sizePayload, float* measures, int sizeMeasureElem, 
                                 const int numMeasures, int* associations, int* isAssoc);

__global__ void createTracks(float* zeroPayload, float* payloads, int sizePayload, float* measures, int sizeMeasureElem, 
                              const int numMeasures, int* isAssoc);
__device__ void createTrack(float* zeroPayload, float* payload, float* measure);

__device__ void matrixAddition(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols);
__device__ void matrixSubtraction(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols);
__device__ void matrixMultiplication(const float A[6][6], const float B[6][6], float C[6][6], const int rowsA, const int colsA, const int colsB);
__device__ void matrixMultTranspose(const float A[6][6], const float T[6][6], float C[6][6], const int rowsA, const int colsA, const int rowsT);

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
constexpr int z_predict_index = 12;
constexpr int S_index = 13;
constexpr int Sinv_index = 14;
constexpr int P_predict_index = 15;
constexpr int x_filter_index = 16;
constexpr int x_predict_index = 17;
constexpr int z_measured_index = 18;
constexpr int bb_size_index = 19;


#endif

