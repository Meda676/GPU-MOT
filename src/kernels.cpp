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

#include <cstdio>

#include "kernels.h"

#define NOT_ASSOCIATED -1

// PREDICT
template <int SIZE>
__device__ void predictKF(float* payload) 
{
    float* p_track_state = payload + dev_offset(track_state_index);
    if (*p_track_state == 2.0) 
    {
        return;
    }

    float* p_first = payload + dev_offset(first_index);
    float* p_x_filter = payload + dev_offset(x_filter_index);
    float* p_x_predict = payload + dev_offset(x_predict_index);
    float* p_z_predict = payload + dev_offset(z_predict_index);
    float* p_F = payload + dev_offset(F_index);
    float* p_G = payload + dev_offset(G_index);
    float* p_Q = payload + dev_offset(Q_index); 
    float* p_S = payload + dev_offset(S_index);
    float* p_H = payload + dev_offset(H_index);
    float* p_P = payload + dev_offset(P_index);
    float* p_P_predict = payload + dev_offset(P_predict_index);
    float* p_R = payload + dev_offset(R_index);

    __shared__ float S[SIZE][SIZE], P_predict[SIZE][SIZE], 
	             x_predict[SIZE], z_predict[SIZE];
    __shared__ float F[SIZE][SIZE], P[SIZE][SIZE], G[SIZE][SIZE], 
	             Q[SIZE][SIZE], R[SIZE][SIZE], H[SIZE][SIZE];
    __shared__ float x_filter[SIZE];

    F[threadIdx.x][threadIdx.y] = p_F[threadIdx.y * SIZE + threadIdx.x];
    P[threadIdx.x][threadIdx.y] = p_P[threadIdx.y * SIZE + threadIdx.x];
    R[threadIdx.x][threadIdx.y] = p_R[threadIdx.y * SIZE + threadIdx.x];
    H[threadIdx.x][threadIdx.y] = p_H[threadIdx.y * SIZE + threadIdx.x];
    S[threadIdx.x][threadIdx.y] = p_S[threadIdx.y * SIZE + threadIdx.x];
    P_predict[threadIdx.x][threadIdx.y] = p_P_predict[threadIdx.y * SIZE + threadIdx.x];
    if (threadIdx.y < 3)
        G[threadIdx.x][threadIdx.y] = p_G[threadIdx.y * SIZE + threadIdx.x];
    if (threadIdx.y < 3 && threadIdx.x < 3)
        Q[threadIdx.x][threadIdx.y] = p_Q[threadIdx.y * 3 + threadIdx.x];
    if (threadIdx.y == 0)
    {
        x_filter[threadIdx.x] = p_x_filter[threadIdx.x];
        x_predict[threadIdx.x] = p_x_predict[threadIdx.x];
        z_predict[threadIdx.x] = p_z_predict[threadIdx.x];
    }
    __syncthreads();

    // Compute Kalman Predict
    if(*p_first == 1.0)
    {
        if ( threadIdx.y == 0)
	{
            x_filter[threadIdx.x] = z_predict[threadIdx.x]; 
            x_predict[threadIdx.x] = x_filter[threadIdx.x];
        }

        *p_first = 0.0;
    }
    else
    {
        matVecMultConst<SIZE>(F, x_filter, x_predict);
    }
    __syncthreads();

    __shared__ float mul1[SIZE][SIZE];
    matrixMultConst<SIZE>(F, P, mul1);
    __syncthreads();
    __shared__ float res1[SIZE][SIZE];
    matrixMultTransposeConst<SIZE>(mul1, F, res1);
    __syncthreads();

    matrixMultiplication(G, Q, mul1, SIZE, 3, 3);
    __syncthreads();
    __shared__ float res2[6][6];
    matrixMultTranspose(mul1, G, res2, SIZE, 3, SIZE);
    __syncthreads();
    matrixAddition<SIZE>(res1, res2, P_predict);
    __syncthreads();

    matrixMultConst<SIZE>(H, P_predict, mul1);
    __syncthreads();
    matrixMultTransposeConst<SIZE>(mul1, H, res1);
    __syncthreads();
    matrixAddition<SIZE>(res1, R, S);

    matVecMultConst<SIZE>(H, x_predict, z_predict);
    __syncthreads();

    p_S[threadIdx.y * SIZE + threadIdx.x] = S[threadIdx.x][threadIdx.y];
    p_P_predict[threadIdx.y * SIZE + threadIdx.x] = P_predict[threadIdx.x][threadIdx.y];
    if(threadIdx.y == 0)
    {
        p_x_filter[threadIdx.x] = x_filter[threadIdx.x];
        p_x_predict[threadIdx.x] = x_predict[threadIdx.x];
        p_z_predict[threadIdx.x] = z_predict[threadIdx.x];
    }
}

template <int SIZE>
__global__ void predictKFs(float* payloads, int sizePayload)
{
    predictKF<SIZE>(payloads + blockIdx.x * sizePayload);
}

void predictKFs_kernel(const int gridDim, const dim3 blockDim, float* payloads, int sizePayload)
{
    predictKFs<6><<<gridDim, blockDim>>>(payloads, sizePayload);
}

template <int SIZE>
__device__ void updateKF(float* payload, float* measure, float* residual)
{
    float* p_track_state = payload + dev_offset(track_state_index);
    if (*p_track_state == 2.0) 
    {
        return;
    }

    float* p_K = payload + dev_offset(K_index);
    float* p_x_filter = payload + dev_offset(x_filter_index);
    float* p_x_predict = payload + dev_offset(x_predict_index);
    float* p_z_predict = payload + dev_offset(z_predict_index);
    float* p_P_predict = payload + dev_offset(P_predict_index);
    float* p_P = payload + dev_offset(P_index);
    float* p_H = payload + dev_offset(H_index);
    float* p_Sinv = payload + dev_offset(Sinv_index); 
    float* p_z_measured = payload + dev_offset(z_measured_index);

    __shared__ float K[SIZE][SIZE], P[SIZE][SIZE], H[SIZE][SIZE], z_measured[SIZE], Sinverse[SIZE][SIZE];
    __shared__ float P_predict[SIZE][SIZE], x_filter[SIZE], x_predict[SIZE], z_predict[SIZE];

    K[threadIdx.x][threadIdx.y] = p_K[threadIdx.y * SIZE + threadIdx.x];
    P[threadIdx.x][threadIdx.y] = p_P[threadIdx.y * SIZE + threadIdx.x];
    H[threadIdx.x][threadIdx.y] = p_H[threadIdx.y * SIZE + threadIdx.x];
    Sinverse[threadIdx.x][threadIdx.y] = p_Sinv[threadIdx.y * SIZE + threadIdx.x];
    P_predict[threadIdx.x][threadIdx.y] = p_P_predict[threadIdx.y * SIZE + threadIdx.x];
    if ( threadIdx.y == 0 )
    {
        x_filter[threadIdx.x] = p_x_filter[threadIdx.x];
        x_predict[threadIdx.x] = p_x_predict[threadIdx.x];
        z_predict[threadIdx.x] = p_z_predict[threadIdx.x];
        z_measured[threadIdx.x] = measure[threadIdx.x];
    }
    __syncthreads();

    __shared__ float res1[SIZE][SIZE];
    matrixMultTransposeConst<SIZE>(P_predict, H, res1);
    __syncthreads();
    matrixMultConst<SIZE>(res1, Sinverse, K);

    __shared__ float res1v[SIZE];
    __shared__ float res2v[SIZE];
    vectorSubtractionConst<SIZE>(z_measured, z_predict, res1v);
     *residual = res1v[0]*res1v[0] + res1v[2]*res1v[2];
    __syncthreads();

    matVecMultConst<SIZE>(K, res1v, res2v);
    __syncthreads();
    vectorAdditionConst<SIZE>(x_predict, res2v, x_filter);
	
    __shared__ float res2[SIZE][SIZE];
    matrixMultConst<SIZE>(K, H, res1);
    __syncthreads();
    matrixMultConst<SIZE>(res1, P_predict, res2);
    __syncthreads();
    matrixSubtraction<SIZE>(P_predict, res2, P);
    __syncthreads();

    p_K[threadIdx.y * SIZE + threadIdx.x] = K[threadIdx.x][threadIdx.y];
    p_P[threadIdx.y * SIZE + threadIdx.x] = P[threadIdx.x][threadIdx.y];
    if (threadIdx.y == 0)
        p_x_filter[threadIdx.x] = x_filter[threadIdx.x];
}

template <int SIZE>
__global__ void updateKFs(float* payloads, int sizePayload, float* measures, int sizeMeasureElem, int* associations, float* residuals)
{
    if(associations[blockIdx.x] != -1)
    {
        updateKF<SIZE>(payloads + blockIdx.x * sizePayload, measures + associations[blockIdx.x] * sizeMeasureElem, residuals + blockIdx.x);
    }
}

void updateKFs_kernel(const int gridDim, const dim3 blockDim, float* payloads, int sizePayload, float* measures, 
                      int sizeMeasureElem, int* associations, float* residuals)
{
    updateKFs<6><<<gridDim, blockDim>>>(payloads, sizePayload, measures, sizeMeasureElem, associations, residuals);
}

__global__ void calculateDistThreshold(float* payloads, int sizePayload, float* measures, int sizeMeasureElem,
                              const int numMeasures, int* associations, int* isAssoc, float* distThreshold)
{
    const float far = 50.0f * 50.0f;

#if MAX_KF <= 500
    __shared__ float curr_dist[512];
    __shared__ int min_index[512];
#else
    __shared__ float curr_dist[1024];
    __shared__ int min_index[1024];
#endif

    float* payload = payloads + blockIdx.x * sizePayload;
    float* p_z_predict = payload + dev_offset(z_predict_index);
    int is_invalid = (payload[dev_offset(track_state_index)] == 2.0f);

    float* meas = measures + threadIdx.x * sizeMeasureElem;
    if (threadIdx.x < numMeasures)
    {
        curr_dist[threadIdx.x] = is_invalid * 1e5f +
                                 (p_z_predict[0]-meas[0])*(p_z_predict[0]-meas[0]) +
                                 (p_z_predict[2]-meas[2])*(p_z_predict[2]-meas[2]) +
                                 (p_z_predict[4]-meas[4])*(p_z_predict[4]-meas[4]);
    }
    else
    {
        curr_dist[threadIdx.x] = far;
    }

    if (curr_dist[threadIdx.x] < far)
    {
        min_index[threadIdx.x] = threadIdx.x;
    }
    else
    {
        curr_dist[threadIdx.x] = far;
        min_index[threadIdx.x] = NOT_ASSOCIATED;
    }
    __syncthreads();

    for (int dist = blockDim.x >> 1; dist >= 2; dist = dist >> 1)
    {
        if (threadIdx.x < dist)
        {
            if (curr_dist[threadIdx.x + dist] < curr_dist[threadIdx.x])
            {
                curr_dist[threadIdx.x] = curr_dist[threadIdx.x + dist];
                min_index[threadIdx.x] = min_index[threadIdx.x + dist];
            }
        }
        if (dist > 16)
            __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (curr_dist[threadIdx.x + 1] < curr_dist[threadIdx.x])
        {
            distThreshold[blockIdx.x] = curr_dist[threadIdx.x + 1];
            associations[blockIdx.x] = min_index[threadIdx.x + 1];
        }
        else
        {
            distThreshold[blockIdx.x] = curr_dist[threadIdx.x];
            associations[blockIdx.x] = min_index[threadIdx.x];
        }

        isAssoc[blockIdx.x] = 0;
    }
}

__global__ void associateAllBIG(float* payloads, int sizePayload, float* measures, 
                                int sizeMeasureElem, const int numMeasures, int* associations, int* isAssoc,
                                float* distThresholdGlob)
{
    if (threadIdx.x >= MAX_KF) 
        return;

    float* payload = payloads + threadIdx.x * sizePayload;   
    
    float* p_track_state = payload + dev_offset(track_state_index);

    float* p_life_time = payload + dev_offset(life_time_index);
    float* p_serial_miss = payload + dev_offset(serial_miss_index);
    float* p_attempt_time = payload + dev_offset(attempt_time_index);

    __shared__ float distThreshold[MAX_KF];
    __shared__ int min_indexes[MAX_KF];

    if (threadIdx.y == 0)
    {
        distThreshold[threadIdx.x] = distThresholdGlob[threadIdx.x];
        min_indexes[threadIdx.x] = 0;
    }
    __syncthreads();

    bool to_kill = false;
    for (int yIdx = threadIdx.y; yIdx < MAX_KF; yIdx += blockDim.y)
    {
        to_kill = to_kill || ((associations[threadIdx.x] != NOT_ASSOCIATED) &&
            (threadIdx.x != yIdx) &&
            (associations[threadIdx.x] == associations[yIdx]) &&
                ((distThreshold[threadIdx.x] > distThreshold[yIdx]) ||
                (distThreshold[threadIdx.x] == distThreshold[yIdx]) && (threadIdx.x > yIdx)));
    }
    __syncthreads();
    
    if (to_kill) 
        associations[threadIdx.x] = NOT_ASSOCIATED;
    __syncthreads();
    
    if ((threadIdx.y == 0) && (associations[threadIdx.x] != NOT_ASSOCIATED))
    {
        min_indexes[associations[threadIdx.x]] = 1;
    }
    __syncthreads();
    
    if (threadIdx.y == 0)
    {
        isAssoc[threadIdx.x] = min_indexes[threadIdx.x];

        if (associations[threadIdx.x] != NOT_ASSOCIATED)
        {
            *p_serial_miss = 0.0f;
            *p_life_time += 1.0f; 
        }
        else 
        {
            *p_serial_miss += 1.0f;
        }
        if (*p_track_state == 0.0f) 
                ++(*p_attempt_time);
    
        if(*p_track_state == 0.0f && *p_life_time >= minDetection)
        {
            *p_track_state = 1.0f; 
            *p_attempt_time = 0;
        }
        else if (*p_track_state == 0.0f && *p_attempt_time >= maxLifeDuration)
        {
            *p_track_state = 2.0f; 
        }
        else if (*p_serial_miss >= maxNotDetection)
        {
            *p_track_state = 2.0f; 
        }
    }
}

__global__ void associateAll(float* payloads, int sizePayload, float* measures, 
                                int sizeMeasureElem, const int numMeasures, int* associations, int* isAssoc)
{
#if MAX_KF <= 100
    if (threadIdx.x >= MAX_KF) 
        return;

    if (threadIdx.y == 0)
        isAssoc[threadIdx.x] = 0;
    __syncthreads();
    
    float* payload = payloads + threadIdx.x * sizePayload;
    
    __shared__ float distThreshold[MAX_KF]; 
    
    float* p_track_state = payload + dev_offset(track_state_index);
    int is_invalid = (*p_track_state == 2.0f);

    float* p_z_predict = payload + dev_offset(z_predict_index);       
    float* p_life_time = payload + dev_offset(life_time_index);
    float* p_serial_miss = payload + dev_offset(serial_miss_index);
    float* p_attempt_time = payload + dev_offset(attempt_time_index);
        
    __shared__ float curr_dist[MAX_KF][MAX_KF];
    __shared__ int min_indexes[8][MAX_KF];
    for (int yIdx = threadIdx.y; yIdx < numMeasures; yIdx += blockDim.y) 
    {
        float* meas = measures + yIdx * sizeMeasureElem;
        curr_dist[yIdx][threadIdx.x] = is_invalid * 1e5f +
		                               (p_z_predict[0]-meas[0])*(p_z_predict[0]-meas[0]) + 
                                       (p_z_predict[2]-meas[2])*(p_z_predict[2]-meas[2]) + 
                                       (p_z_predict[4]-meas[4])*(p_z_predict[4]-meas[4]);
    }
    __syncthreads();

    float local_min = 50.0f * 50.0f;
    int local_min_idx = NOT_ASSOCIATED;
    for (int yIdx = threadIdx.y; yIdx < numMeasures; yIdx += blockDim.y)
    {
        if (curr_dist[yIdx][threadIdx.x] < local_min)
        {
            local_min = curr_dist[yIdx][threadIdx.x];
            local_min_idx = yIdx;
        }
    }
    curr_dist[threadIdx.y][threadIdx.x] = local_min;
    min_indexes[threadIdx.y][threadIdx.x] = local_min_idx;
    __syncthreads();
    if (threadIdx.y < 4)
    {
        if (curr_dist[threadIdx.y + 4][threadIdx.x] < curr_dist[threadIdx.y][threadIdx.x])
	    {
	        curr_dist[threadIdx.y][threadIdx.x] = curr_dist[threadIdx.y + 4][threadIdx.x];
	        min_indexes[threadIdx.y][threadIdx.x] = min_indexes[threadIdx.y + 4][threadIdx.x];
	    }
    }
    __syncthreads();
    if (threadIdx.y < 2)
    {
        if (curr_dist[threadIdx.y + 2][threadIdx.x] < curr_dist[threadIdx.y][threadIdx.x])
        {
            curr_dist[threadIdx.y][threadIdx.x] = curr_dist[threadIdx.y + 2][threadIdx.x];
            min_indexes[threadIdx.y][threadIdx.x] = min_indexes[threadIdx.y + 2][threadIdx.x];
        }
    }
    __syncthreads();
    if (threadIdx.y < 1)
    {
        if (curr_dist[threadIdx.y][threadIdx.x] < curr_dist[threadIdx.y + 1][threadIdx.x])
        {
            distThreshold[threadIdx.x] = curr_dist[threadIdx.y][threadIdx.x];
            associations[threadIdx.x] = min_indexes[threadIdx.y][threadIdx.x];
        }
        else
        {
            distThreshold[threadIdx.x] = curr_dist[threadIdx.y + 1][threadIdx.x];
            associations[threadIdx.x] = min_indexes[threadIdx.y + 1][threadIdx.x];
	    }
        min_indexes[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    bool to_kill = false;
    for (int yIdx = threadIdx.y; yIdx < MAX_KF; yIdx += blockDim.y)
    {
        to_kill = to_kill || ((associations[threadIdx.x] != NOT_ASSOCIATED) &&
            (threadIdx.x != yIdx) &&
            (associations[threadIdx.x] == associations[yIdx]) &&
                ((distThreshold[threadIdx.x] > distThreshold[yIdx]) ||
                (distThreshold[threadIdx.x] == distThreshold[yIdx]) && (threadIdx.x > yIdx)));
    }
    __syncthreads();
    
    if (to_kill)
        associations[threadIdx.x] = NOT_ASSOCIATED;
    __syncthreads();
    
    if ((threadIdx.y == 0) && (associations[threadIdx.x] != NOT_ASSOCIATED))
    {
        min_indexes[threadIdx.y][associations[threadIdx.x]] = 1;
    }
    __syncthreads();
    
    if (threadIdx.y == 0)
    {
        isAssoc[threadIdx.x] = min_indexes[threadIdx.y][threadIdx.x];

        if (associations[threadIdx.x] != NOT_ASSOCIATED)
        {
            *p_serial_miss = 0.0f;
            *p_life_time += 1.0f; 
        }
        else 
        {
           *p_serial_miss += 1.0f;
        }
        if (*p_track_state == 0.0f)
                ++(*p_attempt_time);
    
        if(*p_track_state == 0.0f && *p_life_time >= minDetection)
        {
            *p_track_state = 1.0f; 
            *p_attempt_time = 0;
        }
        else if (*p_track_state == 0.0f && *p_attempt_time >= maxLifeDuration)
        {
            *p_track_state = 2.0f; 
        }
        else if (*p_serial_miss >= maxNotDetection)
        {
            *p_track_state = 2.0f; 
        }
    }
#endif
}

template <int SIZE>
__device__ void createTrack(float* zeroPayload, float* payload, float* measure, int* globalID)
{
    float* p_F = payload + dev_offset(F_index);
    float* p_H = payload + dev_offset(H_index);
    float* p_P = payload + dev_offset(P_index);
    float* p_G = payload + dev_offset(G_index);
    float* p_Q = payload + dev_offset(Q_index); 
    float* p_R = payload + dev_offset(R_index);

    float* p_zeroF = zeroPayload + dev_offset(F_index);
    float* p_zeroH = zeroPayload + dev_offset(H_index);
    float* p_zeroP = zeroPayload + dev_offset(P_index);
    float* p_zeroG = zeroPayload + dev_offset(G_index);
    float* p_zeroQ = zeroPayload + dev_offset(Q_index);
    float* p_zeroR = zeroPayload + dev_offset(R_index);

    if (threadIdx.x < SIZE && threadIdx.y < SIZE)
    {
        p_F[threadIdx.y * SIZE + threadIdx.x] = p_zeroF[threadIdx.y * SIZE + threadIdx.x];
        p_H[threadIdx.y * SIZE + threadIdx.x] = p_zeroH[threadIdx.y * SIZE + threadIdx.x];
        p_P[threadIdx.y * SIZE + threadIdx.x] = p_zeroP[threadIdx.y * SIZE + threadIdx.x];
        p_Q[threadIdx.y * SIZE + threadIdx.x] = p_zeroQ[threadIdx.y * SIZE + threadIdx.x];
        p_R[threadIdx.y * SIZE + threadIdx.x] = p_zeroR[threadIdx.y * SIZE + threadIdx.x];
    }
    if (threadIdx.x < SIZE && threadIdx.y < 3)
        p_G[threadIdx.y * SIZE + threadIdx.x] = p_zeroG[threadIdx.y * SIZE + threadIdx.x];
    __syncthreads();

    if (threadIdx.y == 0)
    {
        *(payload + dev_offset(z_predict_index) + threadIdx.x) = measure[threadIdx.x];
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        *(payload + dev_offset(first_index)) = 1.0;
        *(payload + dev_offset(life_time_index)) = 1.0;
        *(payload + dev_offset(serial_miss_index)) = 0.0;
        *(payload + dev_offset(attempt_time_index)) = 1.0;

        *(payload + dev_offset(bb_size_index)) = 10; 
        *(payload + dev_offset(bb_size_index)+1) = 12; 
        *(payload + dev_offset(bb_size_index)+2) = 0;

        *(int*)(payload + dev_offset(track_ID_index)) = atomicAdd(globalID, 1);
    }
}

template <int SIZE>
__global__ void createTracks(float* zeroPayload, float* payloads, int sizePayload, float* measures, int sizeMeasureElem, const int numMeasures, int* isAssoc, int* globalID)
{
    if(isAssoc[blockIdx.x])
        return;

    float invalid = 2.0f;
    unsigned int invalid_i = *(unsigned int*)&invalid;
    unsigned int taken_i = 0u;
    __shared__ unsigned int outcome;

    float* meas = measures + blockIdx.x * sizeMeasureElem;

    for (int payloadIdx = 0; payloadIdx < MAX_KF; ++payloadIdx)
    {
        float* payload = payloads + ((payloadIdx + blockIdx.x) % MAX_KF) * sizePayload;
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            outcome = atomicCAS((unsigned int*)(payload + dev_offset(track_state_index)), invalid_i, taken_i);
        }
        __syncthreads();
        if (outcome == invalid_i)
        {
            createTrack<SIZE>(zeroPayload, payload, meas, globalID);
            break;
        }
    }
}

void createTracks_kernel(const int gridDim, const dim3 blockDim, float* zeroPayload, float* payloads, int sizePayload,
		         float* measures, int sizeMeasureElem, const int numMeasures, int* isAssoc, int* globalID)
{
	createTracks<6><<<gridDim, blockDim>>>(zeroPayload, payloads, sizePayload, measures, sizeMeasureElem,
			                       numMeasures, isAssoc, globalID);
}

__device__ void matrixMultTranspose(const float A[6][6], const float T[6][6], float C[6][6], const int rowsA, const int colsA, const int rowT)
{
    float cvalue = 0;
    if (threadIdx.y < rowsA && threadIdx.x < rowT)
    {
        for (int k = 0; k < colsA; k++) 
        {
            cvalue += A[threadIdx.y][k] * T[threadIdx.x][k];  
        }

        C[threadIdx.y][threadIdx.x] = cvalue;
    }
}

__device__ void matrixAddition(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols)
{
    if (threadIdx.y < rows && threadIdx.x < cols)
    {
        C[threadIdx.y][threadIdx.x] = A[threadIdx.y][threadIdx.x] + B[threadIdx.y][threadIdx.x];
    }
}

__device__ void matrixSubtraction(const float A[6][6], const float B[6][6], float C[6][6], const int rows, const int cols)
{
    if (threadIdx.y < rows && threadIdx.x < cols)
    {
        C[threadIdx.y][threadIdx.x] = A[threadIdx.y][threadIdx.x] - B[threadIdx.y][threadIdx.x];
    }
}

__device__ void matrixMultiplication(const float A[6][6], const float B[6][6], float C[6][6], const int rowsA, const int colsA, const int colsB)
{
    float cvalue = 0;
    if (threadIdx.y < rowsA && threadIdx.x < colsB)
    {
        for (int k = 0; k < colsA; k++) 
        {
            cvalue += A[threadIdx.y][k] * B[k][threadIdx.x];
        }

        C[threadIdx.y][threadIdx.x] = cvalue;
    }
}
