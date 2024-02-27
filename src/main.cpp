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

#include <iostream>
#include <string>
#include <chrono>
#include <thread>  
#include <cstdarg>
#include <map>
#include <vector>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>	
#include "tracker.h"
#include "detection.h"

bool SHOW_SCREEN;

cv::Mat mosaic(const cv::Mat& img1, const cv::Mat& img2)
{
	cv::Mat _mosaic(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
	img1.copyTo(_mosaic(cv::Rect(0, 0, img1.cols, img1.rows)));
	img2.copyTo(_mosaic(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
	return _mosaic;
}

std::map<int,std::vector<std::vector<std::string>>> readerGT(const std::string& _gt)
{
	std::ifstream file(_gt);
	if(!file.is_open())
	{
		std::cerr << "Error: cannot read the file: " << _gt << std::endl;
		exit(-1);
	}
	
	std::string line;
	std::string delimiter(",");
	std::map<int,std::vector<std::vector<std::string>>> pets;

	while (std::getline(file, line))
	{
		if(line.size() == 0) 
			continue;
		auto start = 0U;
		auto end = line.find(delimiter);
		std::vector<std::string> row;
		while (end != std::string::npos)
		{
			row.push_back(line.substr(start, end - start));
			start = end + delimiter.length();
			end = line.find(delimiter, start);
		}
		row.push_back(line.substr(start, end - start));
		
		const int time = atoi(row[0].c_str()); 
		const int n = atoi(row[1].c_str());
		std::vector<std::vector<std::string>> detections;
		uint j = 2;
		for(uint i = 0; i < n; ++i)
		{
			std::vector<std::string> currDetections;
			try
			{
				currDetections.push_back(row[j]);	
				currDetections.push_back(row[++j]);	
				currDetections.push_back(row[++j]);	
				currDetections.push_back(row[++j]);	
				currDetections.push_back(row[++j]);	
				currDetections.push_back(row[++j]);		
				currDetections.push_back(row[++j]);	
				currDetections.push_back(row[++j]);	
			}
			catch(...)
			{
				std::cerr << "Error: cannot read parse:\n " << line << std::endl;
				exit(-1);
			}
			++j;
			detections.push_back(currDetections);
		}

		pets.insert(std::make_pair(time, detections));
	}

	return pets;
} 


int main(int argc, char** argv)
{
	if (argc < 3) 
	{
		std::cerr << "Usage: " << argv[0] << " <scenario_file> <show_screen>" << std::endl;
		std::cerr << "  <scenario_file>     scenario file path, see README for format details" << std::endl; 
		std::cerr << "  <show_screen>       1=show scenario, 0=don't" << std::endl; 
		return 1;
   }
	
	TrackerParam params;
	params.read(std::string("config/params.txt"));	
	Tracker tracker(params);
	std::vector<Detection> dets;
	cv::Mat image, trackingImg;
	const double millisec = 1000 / 7;
	int img_width=1080;
	int img_height=1920;
	cv::Rect rect;
	std::vector<std::vector<std::string>> curr;
	std::map<int, std::vector<std::vector<std::string>>> detections = readerGT(std::string(argv[1]));
	SHOW_SCREEN = atoi(argv[2]);

	for(uint i=0; i < detections.size(); ++i)
	{
		image = cv::Mat(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));
		curr = detections[i];	
		std::stringstream ss;

		for(auto j=0; j<curr.size(); ++j)
		{
			int label = atoi(curr[j][0].c_str());
			int idx = atoi(curr[j][1].c_str());
			int px = atoi(curr[j][2].c_str());
			int py = atoi(curr[j][3].c_str());
			float vx = atof(curr[j][4].c_str());
			float vy = atof(curr[j][5].c_str());
			int width = atoi(curr[j][6].c_str());
			int height = atoi(curr[j][7].c_str());
			Detection d(Eigen::Vector3f(px,py,0), Eigen::Vector3f(width,height,0), label);
			d.setVelocity(Eigen::Vector3f(vx,vy,0));
			dets.push_back(d);

			if (SHOW_SCREEN)
			{
				rect = cv::Rect(px, py, width, height);
				cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1 );
				trackingImg = image.clone();
				ss.str("");
				ss << j;
				cv::putText(image, ss.str(), cv::Point(px,py), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			}
		}
		
		tracker.track(dets);

		if (SHOW_SCREEN)
		{
			tracker.drawTracks(trackingImg);
			cv::Mat m = mosaic(image, trackingImg);
			cv::imshow("DataTracker", m);					
			cv::waitKey(cvRound(millisec));	
		}
		dets.clear();
	}

	return 0;
}
