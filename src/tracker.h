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

#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>

#include <cublas_v2.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "detection.h"

#if MAX_KF > 1000
#error "Please, choose MAX_KF <= 1000"
#endif

class Tracker
{
protected:
   typedef Eigen::Matrix<float,6,1> Vector6f;
   typedef std::vector<Detection> Detections;

public:
   Tracker();
   ~Tracker();
   void drawTracks(cv::Mat &_img, int img_width, int img_height) const;
   void track(const Detections& _detections);
   
protected:
   constexpr static uint MAX_ASSOC = MAX_KF;

private:
   bool init_;

   float* dev_payloads;        
   float* dev_zeroPayload;     
   float* dev_measurements;    
   int* dev_associations;      
   float** dev_S_ptrs;         
   float** dev_Sinv_ptrs;      
   float* distThreshold_GLOBAL;

   int* dev_measNotAssoc;
   int* dev_info;
   float* dev_residuals;
   int* dev_globalID;

   constexpr static int num_matrices = 21; 
   int offsets[num_matrices];
   std::size_t sizePayload;
   std::size_t sizeMeasure;
   std::size_t sizeZeroPayload;
   int numPayloadElem;
   int sizeMeasureElem;
   float* S_ptrs[MAX_ASSOC];
	float* Sinv_ptrs[MAX_ASSOC];
   cublasHandle_t handle;
};

#endif
