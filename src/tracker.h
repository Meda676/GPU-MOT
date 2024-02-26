#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <chrono>
#include <thread>
#include <sched.h>
#include <pthread.h>
#include "track.h"
#include "tracker_param.h"
#include "detection.h"
#include "kalman.h"
#include <cublas_v2.h>

struct Object
{
   std::shared_ptr<Track> p_tr;
   std::shared_ptr<Kalman> p_kf;

   Object(std::shared_ptr<Track> t, std::shared_ptr<Kalman> k) : p_tr(std::move(t)), p_kf(std::move(k)) {} 
   ~Object() 
   {
      p_tr.reset();
      p_kf.reset();
   }
};


class Tracker
{
protected:
   typedef std::vector<Eigen::Matrix<float,10,1>> Vectors10f;
   typedef Eigen::Matrix<float,6,1> Vector6f;
   typedef std::vector<Eigen::MatrixXf> Matrices;
   typedef std::vector<Detection> Detections;
   typedef std::vector<bool> VecBool;
   typedef std::vector<Object> Objects;

public:
   Tracker(const TrackerParam& _param);
   ~Tracker();
   void drawTracks(cv::Mat &_img) const;
   void track(const Detections& _detections);
   inline uint size() const { return objects_.size();}

protected:
   constexpr static uint MAX_ASSOC = 100;

private:
   uint trackID_;
   bool init_;
   bool startTracking_;
   TrackerParam param_;
   Objects objects_;
   Vectors10f not_associated_; 
   Vectors10f selected_detections;

   std::vector<bool> associable;
	std::vector<int> updated_tracks;
   std::vector <std::vector<float>> costMat;
   Matrices association_matrices;

   Eigen::MatrixXi q;
	Eigen::MatrixXf betaMatrix;
   Eigen::VectorXf betaVector;

   float* dev_payloads; 
   float* dev_zeroPayload;  
   float* dev_measurements;  
   int* dev_associations;  
   float** dev_S_ptrs;    
   float** dev_Sinv_ptrs;   
   int* dev_measNotAssoc;
   int* dev_info;

   constexpr static int num_matrices = 20; 
   int offsets[num_matrices];
   int rows[num_matrices];
	int cols[num_matrices];
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