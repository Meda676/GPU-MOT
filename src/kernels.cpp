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

#include "kernels.h"
#include <cstdio>

#define MAX_KF 100
#define NOT_ASSOCIATED -1

__constant__ int dev_offsets[20];
__constant__ int dev_rows[20];
__constant__ int dev_cols[20];

__constant__ int maxLifeDuration = 5;
__constant__ int maxNotDetection = 5;
__constant__ int minDetection = 3;

__global__ void predictKFs(float* payloads, int sizePayload)
{
    predictKF(payloads + blockIdx.x * sizePayload);
}


__global__ void updateKFs(float* payloads, int sizePayload, float* measures, int sizeMeasureElem, int* associations)
{
    if(associations[blockIdx.x] != -1)
    {
        updateKF(payloads + blockIdx.x * sizePayload, measures + associations[blockIdx.x] * sizeMeasureElem);
    }
}

__global__ void associateAll(float* payloads, int sizePayload, float* measures, 
                                int sizeMeasureElem, const int numMeasures, int* associations, int* isAssoc)
{
    if (threadIdx.x >= MAX_KF)
        return;

    if (threadIdx.y == 0)
        isAssoc[threadIdx.x] = 0; 
    __syncthreads();
    
    float* payload = payloads + threadIdx.x * sizePayload; 
    
    __shared__ float distThreshold[MAX_KF]; 
    
    float* p_track_state = payload + dev_offsets[track_state_index];
    int is_invalid = (*p_track_state == 2.0f); 
    float* p_z_predict = payload + dev_offsets[z_predict_index];   
    float* p_life_time = payload + dev_offsets[life_time_index];
    float* p_serial_miss = payload + dev_offsets[serial_miss_index];
    float* p_attempt_time = payload + dev_offsets[attempt_time_index];
        
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
            *p_attempt_time++;

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


__device__ void predictKF(float* payload) 
{  
    float* p_track_state = payload + dev_offsets[track_state_index];
    if (*p_track_state == 2.0) 
    {
        return;
    }

    float* p_first = payload + dev_offsets[first_index];
    float* p_x_filter = payload + dev_offsets[x_filter_index];
    float* p_x_predict = payload + dev_offsets[x_predict_index];
    float* p_z_predict = payload + dev_offsets[z_predict_index];
    float* p_F = payload + dev_offsets[F_index];
    float* p_G = payload + dev_offsets[G_index];
    float* p_Q = payload + dev_offsets[Q_index]; 
    float* p_S = payload + dev_offsets[S_index];
    float* p_H = payload + dev_offsets[H_index];
    float* p_P = payload + dev_offsets[P_index];
    float* p_P_predict = payload + dev_offsets[P_predict_index];
    float* p_R = payload + dev_offsets[R_index];

    __shared__ float S[6][6], P_predict[6][6], x_predict[6][6], z_predict[6][6];
    __shared__ float F[6][6], P[6][6], G[6][6], Q[6][6], R[6][6], H[6][6];
    __shared__ float x_filter[6][6];
    if ( threadIdx.x < dev_rows[F_index] && threadIdx.y < dev_cols[F_index])
        F[threadIdx.x][threadIdx.y] = p_F[threadIdx.y * dev_rows[F_index] + threadIdx.x];        
    if ( threadIdx.x < dev_rows[x_filter_index] && threadIdx.y < dev_cols[x_filter_index])
        x_filter[threadIdx.x][threadIdx.y] = p_x_filter[threadIdx.y * dev_rows[x_filter_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[P_index] && threadIdx.y < dev_cols[P_index])
        P[threadIdx.x][threadIdx.y] = p_P[threadIdx.y * dev_rows[P_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[G_index] && threadIdx.y < dev_cols[G_index])
        G[threadIdx.x][threadIdx.y] = p_G[threadIdx.y * dev_rows[G_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[Q_index] && threadIdx.y < dev_cols[Q_index])
        Q[threadIdx.x][threadIdx.y] = p_Q[threadIdx.y * dev_rows[Q_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[R_index] && threadIdx.y < dev_cols[R_index])
        R[threadIdx.x][threadIdx.y] = p_R[threadIdx.y * dev_rows[R_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[H_index] && threadIdx.y < dev_cols[H_index])
        H[threadIdx.x][threadIdx.y] = p_H[threadIdx.y * dev_rows[H_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[S_index] && threadIdx.y < dev_cols[S_index])
        S[threadIdx.x][threadIdx.y] = p_S[threadIdx.y * dev_rows[S_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[P_predict_index] && threadIdx.y < dev_cols[P_predict_index])
        P_predict[threadIdx.x][threadIdx.y] = p_P_predict[threadIdx.y * dev_rows[P_predict_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[x_predict_index] && threadIdx.y < dev_cols[x_predict_index])
        x_predict[threadIdx.x][threadIdx.y] = p_x_predict[threadIdx.y * dev_rows[x_predict_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[z_predict_index] && threadIdx.y < dev_cols[z_predict_index])
        z_predict[threadIdx.x][threadIdx.y] = p_z_predict[threadIdx.y * dev_rows[z_predict_index] + threadIdx.x];
    __syncthreads();

	if(*p_first == 1.0)
	{
        if ( threadIdx.x < dev_rows[x_filter_index] && threadIdx.y < dev_cols[x_filter_index])
        {
            x_filter[threadIdx.x][threadIdx.y] = z_predict[threadIdx.x][threadIdx.y]; 
        }
        if ( threadIdx.x < dev_rows[x_predict_index] && threadIdx.y < dev_cols[x_predict_index])
        {
            x_predict[threadIdx.x][threadIdx.y] = x_filter[threadIdx.x][threadIdx.y];
        }

		*p_first = 0.0;
	}
	else
    {
        matrixMultiplication(F, x_filter, x_predict, dev_rows[F_index], dev_cols[F_index], dev_cols[x_filter_index]);
    }
    __syncthreads();

    __shared__ float mul1[6][6];
    matrixMultiplication(F, P, mul1, dev_rows[F_index], dev_cols[F_index], dev_cols[P_index]);
    __syncthreads();
    __shared__ float res1[6][6];
    matrixMultTranspose(mul1, F, res1, dev_rows[F_index], dev_cols[P_index], dev_rows[F_index]);
    __syncthreads();
    matrixMultiplication(G, Q, mul1, dev_rows[G_index], dev_cols[G_index], dev_cols[Q_index]);
    __syncthreads();
    __shared__ float res2[6][6];
    matrixMultTranspose(mul1, G, res2, dev_rows[G_index], dev_cols[Q_index], dev_rows[G_index]);
    matrixAddition(res1, res2, P_predict, dev_rows[P_predict_index], dev_cols[P_predict_index]);
    __syncthreads();

    matrixMultiplication(H, P_predict, mul1, dev_rows[H_index], dev_cols[H_index], dev_cols[P_predict_index]);
    __syncthreads();
    matrixMultTranspose(mul1, H, res1, dev_rows[H_index], dev_cols[P_predict_index], dev_rows[H_index]);
    __syncthreads();
    matrixAddition(res1, R, S, dev_rows[S_index], dev_cols[S_index]);

	matrixMultiplication(H, x_predict, z_predict, dev_rows[H_index], dev_cols[H_index], dev_cols[x_predict_index]);
    __syncthreads();

    if ( threadIdx.x < dev_rows[x_filter_index] && threadIdx.y < dev_cols[x_filter_index])
        p_x_filter[threadIdx.y * dev_rows[x_filter_index] + threadIdx.x] = x_filter[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[P_predict_index] && threadIdx.y < dev_cols[P_predict_index])
        p_P_predict[threadIdx.y * dev_rows[P_predict_index] + threadIdx.x] = P_predict[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[S_index] && threadIdx.y < dev_cols[S_index])
        p_S[threadIdx.y * dev_rows[S_index] + threadIdx.x] = S[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[z_predict_index] && threadIdx.y < dev_cols[z_predict_index])
        p_z_predict[threadIdx.y * dev_rows[z_predict_index] + threadIdx.x] = z_predict[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[x_predict_index] && threadIdx.y < dev_cols[x_predict_index])
        p_x_predict[threadIdx.y * dev_rows[x_predict_index] + threadIdx.x] = x_predict[threadIdx.x][threadIdx.y];  
    __syncthreads();  
}


__device__ void updateKF(float* payload, float* measure) 
{
    float* p_track_state = payload + dev_offsets[track_state_index];
    if (*p_track_state == 2.0) 
    {
        return;
    }

    float* p_K = payload + dev_offsets[K_index];
    float* p_x_filter = payload + dev_offsets[x_filter_index];
    float* p_x_predict = payload + dev_offsets[x_predict_index];
    float* p_z_predict = payload + dev_offsets[z_predict_index];
    float* p_P_predict = payload + dev_offsets[P_predict_index];
    float* p_P = payload + dev_offsets[P_index];
    float* p_H = payload + dev_offsets[H_index];
    float* p_S = payload + dev_offsets[S_index];
    float* p_Sinv = payload + dev_offsets[Sinv_index]; 
    float* p_z_measured = payload + dev_offsets[z_measured_index];

    __shared__ float K[6][6], P[6][6], H[6][6], S[6][6],z_measured[6][6], Sinverse[6][6];
    __shared__ float P_predict[6][6], x_filter[6][6], x_predict[6][6], z_predict[6][6];

    if ( threadIdx.x < dev_rows[K_index] && threadIdx.y < dev_cols[K_index])
        K[threadIdx.x][threadIdx.y] = p_K[threadIdx.y * dev_rows[K_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[P_index] && threadIdx.y < dev_cols[P_index])
        P[threadIdx.x][threadIdx.y] = p_P[threadIdx.y * dev_rows[P_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[H_index] && threadIdx.y < dev_cols[H_index])
        H[threadIdx.x][threadIdx.y] = p_H[threadIdx.y * dev_rows[H_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[S_index] && threadIdx.y < dev_cols[S_index])
        S[threadIdx.x][threadIdx.y] = p_S[threadIdx.y * dev_rows[S_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[P_predict_index] && threadIdx.y < dev_cols[P_predict_index])
        P_predict[threadIdx.x][threadIdx.y] = p_P_predict[threadIdx.y * dev_rows[P_predict_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[x_filter_index] && threadIdx.y < dev_cols[x_filter_index])
        x_filter[threadIdx.x][threadIdx.y] = p_x_filter[threadIdx.y * dev_rows[x_filter_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[x_predict_index] && threadIdx.y < dev_cols[x_predict_index])
        x_predict[threadIdx.x][threadIdx.y] = p_x_predict[threadIdx.y * dev_rows[x_predict_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[z_predict_index] && threadIdx.y < dev_cols[z_predict_index])
        z_predict[threadIdx.x][threadIdx.y] = p_z_predict[threadIdx.y * dev_rows[z_predict_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[z_measured_index] && threadIdx.y < dev_cols[z_measured_index])  
        z_measured[threadIdx.x][threadIdx.y] = measure[threadIdx.y * dev_rows[z_measured_index] + threadIdx.x];
    if ( threadIdx.x < dev_rows[Sinv_index] && threadIdx.y < dev_cols[Sinv_index])
        Sinverse[threadIdx.x][threadIdx.y] = p_Sinv[threadIdx.y * dev_rows[z_measured_index] + threadIdx.x];
    __syncthreads();

    __shared__ float res1[6][6];
    matrixMultTranspose(P_predict, H, res1, dev_rows[P_predict_index], dev_cols[P_predict_index], dev_rows[H_index]);
    __syncthreads();
    matrixMultiplication(res1, Sinverse, K, dev_rows[P_predict_index], dev_cols[P_predict_index], dev_rows[S_index]);
    __syncthreads();

    __shared__ float res2[6][6];
    matrixSubtraction(z_measured, z_predict, res1, dev_rows[z_measured_index], dev_cols[z_measured_index]);
    __syncthreads();
    matrixMultiplication(K, res1, res2, dev_rows[K_index], dev_cols[K_index], dev_cols[z_measured_index]);
    __syncthreads();
    matrixAddition(x_predict, res2, x_filter, dev_rows[x_filter_index], dev_cols[x_filter_index]);
	
    matrixMultiplication(K, H, res1, dev_rows[K_index], dev_cols[K_index], dev_cols[H_index]);
    __syncthreads();
    matrixMultiplication(res1, P_predict, res2, dev_rows[K_index], dev_cols[H_index], dev_cols[P_predict_index]);
    __syncthreads();
    matrixSubtraction(P_predict, res2, P, dev_rows[P_index], dev_cols[P_index]);
    __syncthreads();

    if ( threadIdx.x < dev_rows[K_index] && threadIdx.y < dev_cols[K_index])
        p_K[threadIdx.y * dev_rows[K_index] + threadIdx.x] = K[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[P_index] && threadIdx.y < dev_cols[P_index])
        p_P[threadIdx.y * dev_rows[P_index] + threadIdx.x] = P[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[H_index] && threadIdx.y < dev_cols[H_index])
        p_H[threadIdx.y * dev_rows[H_index] + threadIdx.x] = H[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[S_index] && threadIdx.y < dev_cols[S_index])
        p_S[threadIdx.y * dev_rows[S_index] + threadIdx.x] = S[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[P_predict_index] && threadIdx.y < dev_cols[P_predict_index])
        p_P_predict[threadIdx.y * dev_rows[P_predict_index] + threadIdx.x] = P_predict[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[x_filter_index] && threadIdx.y < dev_cols[x_filter_index])
        p_x_filter[threadIdx.y * dev_rows[x_filter_index] + threadIdx.x] = x_filter[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[x_predict_index] && threadIdx.y < dev_cols[x_predict_index])
        p_x_predict[threadIdx.y * dev_rows[x_predict_index] + threadIdx.x] = x_predict[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[z_predict_index] && threadIdx.y < dev_cols[z_predict_index])
        p_z_predict[threadIdx.y * dev_rows[z_predict_index] + threadIdx.x] = z_predict[threadIdx.x][threadIdx.y];
    if ( threadIdx.x < dev_rows[z_measured_index] && threadIdx.y < dev_cols[z_measured_index])
        p_z_measured[threadIdx.y * dev_rows[z_measured_index] + threadIdx.x] = z_measured[threadIdx.x][threadIdx.y];
    __syncthreads();
}


__global__ void createTracks(float* zeroPayload, float* payloads, int sizePayload, float* measures, int sizeMeasureElem, const int numMeasures, int* isAssoc)  
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
            outcome = atomicCAS((unsigned int*)(payload + dev_offsets[track_state_index]), invalid_i, taken_i);
         }
        __syncthreads();
        if (outcome == invalid_i)
        {
                createTrack(zeroPayload, payload, meas);
            break;
        }
    }
}

__device__ void createTrack(float* zeroPayload, float* payload, float* measure)
{
    float* p_F = payload + dev_offsets[F_index];
    float* p_H = payload + dev_offsets[H_index];
    float* p_P = payload + dev_offsets[P_index];
    float* p_G = payload + dev_offsets[G_index];
    float* p_Q = payload + dev_offsets[Q_index]; 
    float* p_R = payload + dev_offsets[R_index];

    float* p_zeroF = zeroPayload + dev_offsets[F_index];
    float* p_zeroH = zeroPayload + dev_offsets[H_index];
    float* p_zeroP = zeroPayload + dev_offsets[P_index];
    float* p_zeroG = zeroPayload + dev_offsets[G_index];
    float* p_zeroQ = zeroPayload + dev_offsets[Q_index];
    float* p_zeroR = zeroPayload + dev_offsets[R_index];

    if (threadIdx.x < dev_rows[F_index] && threadIdx.y < dev_cols[F_index])
        p_F[threadIdx.y * dev_rows[F_index] + threadIdx.x] = p_zeroF[threadIdx.y * dev_rows[F_index] + threadIdx.x];
    if (threadIdx.x < dev_rows[H_index] && threadIdx.y < dev_cols[H_index])
        p_H[threadIdx.y * dev_rows[H_index] + threadIdx.x] = p_zeroH[threadIdx.y * dev_rows[H_index] + threadIdx.x];
    if (threadIdx.x < dev_rows[P_index] && threadIdx.y < dev_cols[P_index])
        p_P[threadIdx.y * dev_rows[P_index] + threadIdx.x] = p_zeroP[threadIdx.y * dev_rows[P_index] + threadIdx.x];
    if (threadIdx.x < dev_rows[G_index] && threadIdx.y < dev_cols[G_index])
        p_G[threadIdx.y * dev_rows[G_index] + threadIdx.x] = p_zeroG[threadIdx.y * dev_rows[G_index] + threadIdx.x];
    if (threadIdx.x < dev_rows[Q_index] && threadIdx.y < dev_cols[Q_index])
        p_Q[threadIdx.y * dev_rows[Q_index] + threadIdx.x] = p_zeroQ[threadIdx.y * dev_rows[Q_index] + threadIdx.x];
    if (threadIdx.x < dev_rows[R_index] && threadIdx.y < dev_cols[R_index])
        p_R[threadIdx.y * dev_rows[R_index] + threadIdx.x] = p_zeroR[threadIdx.y * dev_rows[R_index] + threadIdx.x];
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        *(payload + dev_offsets[first_index]) = 1.0;
        *(payload + dev_offsets[life_time_index]) = 1.0;
        *(payload + dev_offsets[serial_miss_index]) = 0.0;
        *(payload + dev_offsets[attempt_time_index]) = 1.0;

        *(payload + dev_offsets[z_predict_index]) = *measure; 
        *(payload + dev_offsets[z_predict_index]+1) = *(measure+1); 
        *(payload + dev_offsets[z_predict_index]+2) = *(measure+2); 
        *(payload + dev_offsets[z_predict_index]+3) = *(measure+3); 
        *(payload + dev_offsets[z_predict_index]+4) = *(measure+4); 
        *(payload + dev_offsets[z_predict_index]+5) = *(measure+5); 

        *(payload + dev_offsets[bb_size_index]) = 10; 
        *(payload + dev_offsets[bb_size_index]+1) = 12; 
        *(payload + dev_offsets[bb_size_index]+2) = 0;
    }
    __syncthreads();
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
