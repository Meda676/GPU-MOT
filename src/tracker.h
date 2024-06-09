#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <numeric>
#include <chrono>
#include <thread>
#include <sched.h>
#include <pthread.h>
#include "detection.h"
#include <cublas_v2.h>

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

   constexpr static int num_matrices = 20; 
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
