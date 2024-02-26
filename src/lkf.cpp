#include "lkf.h"
#include "kernels.h"

KF_linear::KF_linear(const float& dt, const Eigen::Vector3f& target_delta, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, 
                     const Eigen::Vector3f& dim, const Eigen::MatrixXf& _R, const Eigen::Matrix3f& _Q, const int _label) : Kalman(dt, target_delta, pos, vel, dim, _R, _Q, _label)
{
	innov_combined = Eigen::VectorXf::Zero(measure_size);
	innov_doub = Eigen::MatrixXf::Zero(measure_size,measure_size);
	P_temp = Eigen::MatrixXf::Zero(state_size, state_size);
	size_filter = Eigen::Vector3f(0,0,0);
}

KF_linear::~KF_linear()
{

}

void KF_linear::retrieveUpdate(float* payload, int* offsets)
{
	float* z_predict_ptr2 = payload + offsets[z_predict_index]; 
	cudaMemcpy(z_predict.data(), z_predict_ptr2, (z_predict.cols()*z_predict.rows()) * sizeof(float), cudaMemcpyDeviceToHost);
}

Eigen::Vector3f KF_linear::predict()
{
	if(first == 1.0)
	{
		x_filter << z_predict(0), z_predict(1), z_predict(2), z_predict(3), z_predict(4), z_predict(5);
		x_predict = x_filter;
		first = 0.0;
	}
	else
	{
		x_predict = F*x_filter;
	}
	
	P_predict = F * P * F.transpose() + G * Q * G.transpose();  
	S = H * P_predict * H.transpose() + R;
	z_predict = H * x_predict;

	return Eigen::Vector3f(z_predict(0), z_predict(2), z_predict(4));
}


Eigen::VectorXf KF_linear::update(const Vector6f& selected_detection, const Eigen::Vector3f& detection_size) 
{		
	z_measured = selected_detection;
	bb_size = detection_size;
	
	K = P_predict * H.transpose() * S.inverse();	
	x_filter = x_predict + K * (z_measured - z_predict);		
	P = P_predict - K * H * P_predict;
	
	return x_filter;
}