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

#include "tracker.h"
#include "kernels.h"
#include <cassert>

extern __constant__ int dev_offsets[20];
extern __constant__ int dev_rows[20];
extern __constant__ int dev_cols[20];

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

Tracker::Tracker(const TrackerParam &_param) : param_(_param)
{
	trackID_ = 0;
	init_ = false;
	startTracking_ = false;

	float dt = 0.1; 
	Eigen::MatrixXf F = Eigen::MatrixXf(6, 6);
	F << 1, dt, 0, 0, 0, 0,
		  0, 1, 0, 0, 0, 0,
		  0, 0, 1, dt, 0, 0,
	     0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 1, dt,
		  0, 0, 0, 0, 0, 1;
	Eigen::MatrixXf H = Eigen::MatrixXf(6, 6);
	H << 1, 0, 0, 0, 0, 0,
		  0, 1, 0, 0, 0, 0,  
		  0, 0, 1, 0, 0, 0,
		  0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 1, 0,
		  0, 0, 0, 0, 0, 1;
	Eigen::Vector3f target_delta(10,10,10); 
	Eigen::MatrixXf P = Eigen::MatrixXf(6, 6);
	P << target_delta(0), 0, 0, 0, 0, 0, 
		  0, 1, 0, 0, 0, 0, 
		  0, 0, target_delta(1), 0, 0, 0,
		  0, 0, 0, 1, 0, 0, 
		  0, 0, 0, 0, target_delta(2), 0,
		  0, 0, 0, 0, 0, 1;
	Eigen::MatrixXf G = Eigen::MatrixXf(6, 3);
	G << std::pow(dt, 2)/2, 0, 0,
		  dt, 0, 0,
		  0, std::pow(dt, 2)/2, 0,
		  0, dt, 0,
		  0, 0, std::pow(dt, 2)/2,
		  0, 0, dt;
	Eigen::MatrixXf Q = Eigen::MatrixXf(3, 3);
	Q << 500, 0, 0,
		  0, 500, 0,
        0, 0, 500;
	Eigen::MatrixXf R = Eigen::MatrixXf(6, 6);
	R << 3, 0, 0, 0, 0, 0,
		  0, 3, 0, 0, 0, 0,
		  0, 0, 3, 0, 0, 0,
		  0, 0, 0, 3, 0, 0,
		  0, 0, 0, 0, 3, 0,
		  0, 0, 0, 0, 0, 3;
	Eigen::MatrixXf S = Eigen::MatrixXf(6, 6);
	S << 1, 0, 0, 0, 0, 0,
		  0, 1, 0, 0, 0, 0,
		  0, 0, 1, 0, 0, 0,
		  0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 1, 0,
		  0, 0, 0, 0, 0, 1;
	Eigen::MatrixXf K = Eigen::MatrixXf(6, 6);
	Eigen::VectorXf x_filter = Eigen::VectorXf(6); 
	Eigen::VectorXf x_predict = Eigen::VectorXf(6);
	Eigen::VectorXf z_predict = Eigen::VectorXf(6); 
	Eigen::VectorXf z_measured = Eigen::VectorXf(6);
	Eigen::Vector3f bb_size = Eigen::Vector3f(10,12,0); 

	offsets[F_index] = 0;
	offsets[H_index] = offsets[F_index] + F.cols() * F.rows();
	offsets[P_index] = offsets[H_index] + H.cols() * H.rows();
	offsets[G_index] = offsets[P_index] + P.cols() * P.rows();
	offsets[Q_index] = offsets[G_index] + G.cols() * G.rows();
	offsets[R_index] = offsets[Q_index] + Q.cols() * Q.rows();
	offsets[first_index] = offsets[R_index] + R.cols() * R.rows();
	offsets[life_time_index] = offsets[first_index] + 1 * 1;
	offsets[track_state_index] = offsets[life_time_index] + 1 * 1;
	offsets[serial_miss_index] = offsets[track_state_index] + 1 * 1;
	offsets[attempt_time_index] = offsets[serial_miss_index] + 1 * 1;
	offsets[K_index] = offsets[attempt_time_index]  + 1 * 1;
	offsets[S_index] = offsets[K_index] + K.cols() * K.rows();
	offsets[Sinv_index] = offsets[S_index] + S.cols() * S.rows();
	offsets[P_predict_index] = offsets[Sinv_index] + S.cols() * S.rows();
	offsets[x_filter_index] = offsets[P_predict_index] + P.cols() * P.rows();
	offsets[x_predict_index] = offsets[x_filter_index] + x_filter.cols() * x_filter.rows();
	offsets[z_predict_index] = offsets[x_predict_index] + x_predict.cols() * x_predict.rows();
	offsets[z_measured_index] = offsets[z_predict_index] + z_predict.cols() * z_predict.rows();
	offsets[bb_size_index] = offsets[z_measured_index] + 6 * 1;
	
	rows[F_index] = F.rows();
	cols[F_index] = F.cols();
	rows[H_index] = H.rows();
	cols[H_index] = H.cols();
	rows[P_index] = P.rows();
	cols[P_index] = P.cols();
	rows[G_index] = G.rows();
	cols[G_index] = G.cols();
	rows[Q_index] = Q.rows();
	cols[Q_index] = Q.cols();
	rows[R_index] = R.rows();
	cols[R_index] = R.cols();
	rows[first_index] = 1;
	cols[first_index] = 1;
	rows[life_time_index] = 1;
	cols[life_time_index] = 1;
	rows[track_state_index] = 1;
	cols[track_state_index] = 1;
	rows[serial_miss_index] = 1;
	cols[serial_miss_index] = 1;
	rows[attempt_time_index] = 1;
	cols[attempt_time_index] = 1;
	rows[K_index] = K.rows();
	cols[K_index] = K.cols();
	rows[S_index] = S.rows();
	cols[S_index] = S.cols();
	rows[Sinv_index] = S.rows();
	cols[Sinv_index] = S.cols(); 
	rows[P_predict_index] = P.rows();
	cols[P_predict_index] = P.cols();
	rows[x_filter_index] = x_filter.rows();
	cols[x_filter_index] = x_filter.cols();
	rows[x_predict_index] = x_predict.rows();
	cols[x_predict_index] = x_predict.cols();
	rows[z_predict_index] = z_predict.rows();
	cols[z_predict_index] = z_predict.cols();
	rows[z_measured_index] = z_measured.rows();
	cols[z_measured_index] = z_measured.cols();
	rows[bb_size_index] = bb_size.rows();
	cols[bb_size_index] = bb_size.cols();

	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_offsets, &offsets, num_matrices*sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_rows, &rows, num_matrices*sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_cols, &cols, num_matrices*sizeof(int)));

	numPayloadElem = (
		F.cols() * F.rows() +
		H.cols() * H.rows() +
		P.cols() * P.rows() +
		G.cols() * G.rows() +
		Q.cols() * Q.rows() +
		R.cols() * R.rows() +
		K.cols() * K.rows() +
		S.cols() * S.rows() +
		S.cols() * S.rows() +
		P.cols() * P.rows() +  
		x_filter.cols() * x_filter.rows() +
		x_predict.cols() * x_predict.rows() +
		z_predict.cols() * z_predict.rows() +
		z_measured.cols()* z_measured.rows() + 
		bb_size.cols()* bb_size.rows()	 		
   ) + 1 	
	  + 4; 

	sizePayload = (numPayloadElem) * sizeof(float);

	sizeZeroPayload = sizeof(float) * (
		F.cols() * F.rows() +
		H.cols() * H.rows() +
		P.cols() * P.rows() +
		G.cols() * G.rows() +
		Q.cols() * Q.rows() +
		R.cols() * R.rows() + 
		5 
	); 

	sizeMeasureElem = 6;
	sizeMeasure = sizeof(float) * sizeMeasureElem;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_payloads, MAX_ASSOC * sizePayload));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_zeroPayload, sizeZeroPayload));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_measNotAssoc, MAX_ASSOC * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_measurements, MAX_ASSOC * sizeMeasure));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_associations, MAX_ASSOC * sizeof(int)));

	size_t so_far = 0u;
	float firstInitVal = 1.0;
	float life_time = 1.0;
	float track_state = 2.0;
	float serial_miss = 0.0;
	float attempt_time = 1.0;
	
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload, F.data(), F.cols()*F.rows()*sizeof(float), cudaMemcpyHostToDevice));
	so_far += F.rows() * F.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, H.data(), H.cols()*H.rows()*sizeof(float), cudaMemcpyHostToDevice));
	so_far += H.rows() * H.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, P.data(), P.cols()*P.rows()*sizeof(float), cudaMemcpyHostToDevice));
	so_far += P.rows() * P.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, G.data(), G.cols()*G.rows()*sizeof(float), cudaMemcpyHostToDevice));
	so_far += G.rows() * G.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, Q.data(), Q.cols()*Q.rows()*sizeof(float), cudaMemcpyHostToDevice));
	so_far += Q.rows() * Q.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, R.data(), R.cols()*R.rows()*sizeof(float), cudaMemcpyHostToDevice));

	so_far += R.rows() * R.cols();
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, &firstInitVal, sizeof(float), cudaMemcpyHostToDevice));
	so_far += 1;
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, &life_time, sizeof(float), cudaMemcpyHostToDevice));
	so_far += 1;
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, &track_state, sizeof(float), cudaMemcpyHostToDevice));
	so_far += 1;
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, &serial_miss, sizeof(float), cudaMemcpyHostToDevice));
	so_far += 1;
	CHECK_CUDA_ERROR(cudaMemcpy(dev_zeroPayload + so_far, &attempt_time, sizeof(float), cudaMemcpyHostToDevice));

	so_far = 0u;
	for (int i=0; i< MAX_ASSOC; ++i)
	{
		CHECK_CUDA_ERROR(cudaMemcpy(dev_payloads + so_far, dev_zeroPayload, sizeZeroPayload, cudaMemcpyDeviceToDevice));
		so_far += numPayloadElem;
	}

	CHECK_CUDA_ERROR(cudaMemset(dev_measNotAssoc, 0, MAX_ASSOC * sizeof(int)));	
	CHECK_CUDA_ERROR(cudaMemset(dev_associations, -1, MAX_ASSOC * sizeof(int))); 

	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_info, MAX_ASSOC * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_S_ptrs, MAX_ASSOC * sizeof(float*)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_Sinv_ptrs, MAX_ASSOC * sizeof(float*)));
	for (int i = 0; i < MAX_ASSOC; ++i) {
		S_ptrs[i] = dev_payloads + i * numPayloadElem + offsets[S_index];
		Sinv_ptrs[i] = dev_payloads + i * numPayloadElem + offsets[Sinv_index];
	} 
	CHECK_CUDA_ERROR(cudaMemcpy(dev_S_ptrs, S_ptrs, MAX_ASSOC*sizeof(float*), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(dev_Sinv_ptrs, Sinv_ptrs, MAX_ASSOC*sizeof(float*), cudaMemcpyHostToDevice));

	cublasCreate(&handle);
}

Tracker::~Tracker()
{
	CHECK_CUDA_ERROR(cudaFree(dev_payloads));
	CHECK_CUDA_ERROR(cudaFree(dev_zeroPayload));
	CHECK_CUDA_ERROR(cudaFree(dev_measNotAssoc));
	CHECK_CUDA_ERROR(cudaFree(dev_measurements));
	CHECK_CUDA_ERROR(cudaFree(dev_associations));
	CHECK_CUDA_ERROR(cudaFree(dev_S_ptrs));
	CHECK_CUDA_ERROR(cudaFree(dev_Sinv_ptrs));
	CHECK_CUDA_ERROR(cudaFree(dev_info));
	cublasDestroy(handle);
}

void Tracker::track(const Tracker::Detections &_detections)
{
	int numMeasures = _detections.size();
	std::vector<Vector6f> measure2cuda(100, Vector6f(-1,-1,-1,-1,-1,-1));
	for (int d = 0; d < _detections.size(); ++d)
	{
		Vector6f detection = Vector6f(_detections[d].x(), _detections[d].vx(), _detections[d].y(),
												_detections[d].vy(), _detections[d].z(), _detections[d].vz());
		
		measure2cuda[d] = detection;
	}
	CHECK_CUDA_ERROR(cudaMemcpy(dev_measurements, measure2cuda.data(), measure2cuda.size() * 6 * sizeof(float), cudaMemcpyHostToDevice));

	if (!init_)
	{
		createTracks<<<numMeasures,dim3(8,8,1)>>>(dev_zeroPayload, dev_payloads, numPayloadElem, dev_measurements, sizeMeasureElem, numMeasures, dev_measNotAssoc);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		init_ = true;
	}
	else
	{		
		predictKFs<<<MAX_ASSOC,dim3(8,8,1)>>>(dev_payloads, numPayloadElem);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		
		associateAll<<<1,dim3(128,8,1)>>>(dev_payloads, numPayloadElem, dev_measurements, sizeMeasureElem, numMeasures, dev_associations, dev_measNotAssoc);

		createTracks<<<numMeasures,dim3(8,8,1)>>>(dev_zeroPayload, dev_payloads, numPayloadElem, dev_measurements, sizeMeasureElem, numMeasures, dev_measNotAssoc);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		
		cublasStatus_t status = cublasSmatinvBatched(handle, rows[S_index], dev_S_ptrs, cols[S_index], dev_Sinv_ptrs, cols[Sinv_index], dev_info, MAX_ASSOC);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		updateKFs<<<MAX_ASSOC,dim3(8,8,1)>>>(dev_payloads, numPayloadElem, dev_measurements, sizeMeasureElem, dev_associations);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}
}


void Tracker::drawTracks(cv::Mat &_img) const
{
	std::vector<std::vector<int>> bb_line;

	int img_width = 1920;
	int img_height = 1080;
	cv::Rect rect;
	cv::Point p2D;
	std::stringstream ss;
	for (int i=0; i<MAX_ASSOC; ++i)
	{
		float track_state;
		Eigen::VectorXf bb_size = Eigen::VectorXf::Zero(3);
		Eigen::VectorXf z_predict = Eigen::VectorXf::Zero(6);
		float* payload = dev_payloads + i * numPayloadElem;
		float* z_predict_ptr = payload + offsets[z_predict_index]; 
		float* track_state_ptr = payload + offsets[track_state_index];	
		float* bb_size_ptr = payload + offsets[bb_size_index]; 
		cudaMemcpy(z_predict.data(), z_predict_ptr, (z_predict.cols()*z_predict.rows()) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&track_state, track_state_ptr, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bb_size.data(), bb_size_ptr, (bb_size.cols()*bb_size.rows()) * sizeof(float), cudaMemcpyDeviceToHost);
		
		if(track_state == 1.0)
		{
			ss.str("");
			ss << i;
			std::flush(ss);
			rect = cv::Rect(z_predict(0), z_predict(2), bb_size(0), bb_size(1));
			p2D = cv::Point(z_predict(0), z_predict(2));

			if (rect.x < img_width && rect.y < img_height)
			{
				cv::rectangle(_img, rect, cv::Scalar(0, 0, 255), 1);
				cv::putText(_img, ss.str(), p2D, cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			}

			std::vector<int> v = {i, (int)z_predict(0), (int)z_predict(2), (int)bb_size(0), (int)bb_size(1)};
			bb_line.push_back(v);
		}
	}
}
