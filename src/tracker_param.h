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