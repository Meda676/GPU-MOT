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