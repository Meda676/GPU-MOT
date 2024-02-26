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

/*** Kalman implementation ***/ 

#include "kalman.h"
#include <cuda_runtime.h>

Kalman::Kalman(const float& dt, const Eigen::Vector3f& target_delta, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, 
               const Eigen::Vector3f& dim, const Eigen::MatrixXf& _R, const Eigen::Matrix3f& _Q, const int _label)
{
	K = Eigen::MatrixXf(state_size, measure_size);
  	x_filter = Eigen::VectorXf::Zero(state_size);    
	F = Eigen::MatrixXf(state_size, state_size);
	F << 1, dt, 0, 0, 0, 0,
		  0, 1, 0, 0, 0, 0,
		  0, 0, 1, dt, 0, 0,
	     0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 1, dt,
		  0, 0, 0, 0, 0, 1;
	H = Eigen::MatrixXf(measure_size, state_size);
	H << 1, 0, 0, 0, 0, 0,
		  0, 1, 0, 0, 0, 0,  
		  0, 0, 1, 0, 0, 0,
		  0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 1, 0,
		  0, 0, 0, 0, 0, 1;
 	P = Eigen::MatrixXf(state_size, state_size);
	P << target_delta(0), 0, 0, 0, 0, 0, 
		  0, 1, 0, 0, 0, 0, 
		  0, 0, target_delta(1), 0, 0, 0,
		  0, 0, 0, 1, 0, 0, 
		  0, 0, 0, 0, target_delta(2), 0,
		  0, 0, 0, 0, 0, 1;
	G = Eigen::MatrixXf(state_size, 3);
	G << std::pow(dt, 2)/2, 0, 0,
		  dt, 0, 0,
		  0, std::pow(dt, 2)/2, 0,
		  0, dt, 0,
		  0, 0, std::pow(dt, 2)/2,
		  0, 0, dt;

  	S = Eigen::MatrixXf::Zero(measure_size,measure_size);
  	Q = _Q;
  	R = _R;
	x_predict = Eigen::VectorXf::Zero(state_size);
	P_predict = Eigen::MatrixXf::Zero(state_size,state_size);
	z_predict = Eigen::VectorXf::Zero(measure_size);
	innov = Eigen::VectorXf::Zero(measure_size);
	z_measured = Eigen::VectorXf::Zero(measure_size);

  	nis = 0;   
	first = 1.0;
	delta_time = dt;
	timestamp = 0;
	label = _label;
	
	bb_size << dim(0), dim(1), dim(2);
	z_predict(0) = pos(0);
	z_predict(2) = pos(1);
	z_predict(4) = pos(2); 

	if (vel(0) < -5000 || vel(1) < -5000 || vel(2) < -5000)
	{
		z_predict(1) = 0.0;
		z_predict(3) = 0.0;
		z_predict(5) = 0.0;
	}
	else
	{
		z_predict(1) = vel(0);
		z_predict(3) = vel(1);
		z_predict(5) = vel(2);
	}
	
}