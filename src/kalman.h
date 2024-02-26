#ifndef _KALMAN_H_
#define _KALMAN_H_

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

class Kalman
{
protected:
   typedef Eigen::Matrix<float,6,1> Vector6f;
   typedef Eigen::Matrix<float,10,1> Vector10f;

public:
   Kalman(const float& dt, const Eigen::Vector3f& target_delta, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, 
            const Eigen::Vector3f& dim, const Eigen::MatrixXf& _R, const Eigen::Matrix3f& _Q, const int _label);
   
   virtual Eigen::Vector3f predict() = 0;
   virtual Eigen::VectorXf update(const Vector6f& selected_detection, const Eigen::Vector3f& detection_size) = 0;
   virtual void retrieveUpdate(float* payload, int* offsets) = 0;

   inline const Eigen::MatrixXf getS() const { return S; }
   inline const Eigen::Vector3f getLastPosition() const { return Eigen::Vector3f(z_predict(0), z_predict(2), z_predict(4)); }
   inline const Eigen::Vector3f getLastSpeed() const { return Eigen::Vector3f(z_predict(1), z_predict(4), z_predict(5)); }
   inline const Eigen::VectorXf getLastPredictionEigen() const { return z_predict; }
   inline Eigen::Vector3f getBBsize() const { return bb_size; }
   inline int getLabel() const { return label; }
   inline float getFirst() const {return first; }

protected:
   const int measure_size = 6;  
   const int state_size = 6;  

   Eigen::MatrixXf K; 
   Eigen::MatrixXf F; 
   Eigen::MatrixXf H;   
   Eigen::MatrixXf G;  
   Eigen::MatrixXf P;  
   Eigen::MatrixXf S;  
   Eigen::MatrixXf R; 
   Eigen::Matrix3f Q;
   
   Eigen::VectorXf x_filter; 
   Eigen::MatrixXf P_predict;
   Eigen::VectorXf x_predict;
   Eigen::VectorXf z_predict;      
   Eigen::VectorXf innov;      
   Eigen::VectorXf z_measured;  

   float nis;            
   float first;
   int label;
   float timestamp;
   float delta_time;
   Eigen::Vector3f bb_size;  
};

#endif