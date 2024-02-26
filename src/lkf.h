#ifndef _LKF_H_
#define _LKF_H_

#include "kalman.h"

class KF_linear : public Kalman
{
  public:
    KF_linear(const float& dt, const Eigen::Vector3f& target_delta, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, 
              const Eigen::Vector3f& dim, const Eigen::MatrixXf& _R, const Eigen::Matrix3f& _Q, const int _label);
    ~KF_linear();
  
    Eigen::Vector3f predict();
    Eigen::VectorXf update(const Vector6f& selected_detection, const Eigen::Vector3f& detection_size) override;
    void retrieveUpdate(float* payload, int* offsets) override;

  private:
    Eigen::VectorXf innov_combined;
    Eigen::MatrixXf innov_doub;
    Eigen::MatrixXf P_temp;
    Eigen::Vector3f size_filter;
    Eigen::MatrixXf Sinv;
};

#endif