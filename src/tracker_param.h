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

#ifndef _TRACKER_PARAM_H_
#define _TRACKER_PARAM_H_

#include <iostream>
#include <fstream>
#include <Eigen/Core>

class TrackerParam
{
	public:
		float pd = 1;
		float lambda = 2;
		float gamma = lambda * 0.000001;
		float g_sigma;
		float assocCost;
		float maxCost;
		float global_assocCost;
		float global_g_sigma;
		int maxLife;
		int maxNotDet;
		int minDet;
		Eigen::Vector3f target_delta;
		Eigen::MatrixXf R;
		Eigen::MatrixXf Q;
		float dt;
		std::string dataType;

	public:
		TrackerParam() { ; }
		void read(const std::string& filename);
		TrackerParam& operator=(const TrackerParam& param_copy)
		{
			this->pd = param_copy.pd;
			this->g_sigma = param_copy.g_sigma;
			this->lambda = param_copy.lambda;
			this->gamma = param_copy.gamma;
			this->target_delta = param_copy.target_delta;
			this->assocCost = param_copy.assocCost;
			this->global_assocCost = param_copy.global_assocCost;
			this->global_g_sigma = param_copy.global_g_sigma;
			this->maxCost = param_copy.maxCost;
			this->dt = param_copy.dt;
			this->maxLife = param_copy.maxLife;
			this->maxNotDet = param_copy.maxNotDet;
			this->minDet = param_copy.minDet;
			return *this;
		}
};


#endif