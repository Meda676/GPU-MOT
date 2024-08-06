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

#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>    
#include <vector>

#include <experimental/filesystem>
#include <opencv2/opencv.hpp>	

#include "detection.h"
#include "tracker.h"

bool SHOW_SCREEN;
bool DUMP_ON = true;

struct ObjectFile {
    int label;
    int id;
    int posX;
    int posY;
    float velX;
    float velY;
    int width;
    int height;
};

struct Line {
    int timeInstant;
    int numberOfObjects;
    std::vector<ObjectFile> objects;
};

std::chrono::microseconds::rep tracking_time;
std::chrono::microseconds::rep lkf_predict_time, lkf_update_time;
std::chrono::microseconds::rep gnn_time;
std::chrono::microseconds::rep trackUpd_time;

cv::Mat mosaic(const cv::Mat& img1, const cv::Mat& img2)
{
	cv::Mat _mosaic(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
	img1.copyTo(_mosaic(cv::Rect(0, 0, img1.cols, img1.rows)));
	img2.copyTo(_mosaic(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
	return _mosaic;
}

void write_file(std::vector<std::string> data, std::string filename)
{ 
	std::ofstream outFile(filename, std::ios::app);

	if (!outFile.is_open()) 
	{
        std::cerr << "Error: Could not open the file " << filename << " for writing." << std::endl;
        return;
   }

	for (int i=0; i<data.size(); ++i)
		outFile << data[i];

	outFile << std::endl;
	outFile.close();
}

std::vector<Line> readerGT(const std::string& filePath)
{
	std::vector<Line> parsedData;
	std::ifstream file(filePath);
	if(!file.is_open())
	{
		std::cerr << "Error: cannot read the file: " << filePath << std::endl;
		return parsedData;
	}
	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream stream(line);
        	std::string element;
	        Line currentLine;
	        
		std::getline(stream, element, ',');
        	currentLine.timeInstant = std::stoi(element);
        	 std::getline(stream, element, ',');
        	currentLine.numberOfObjects = std::stoi(element);
		//each row = [time, num_det, {det}] ;; each det = [label, px, py, vx, vy, width, height] 
		for (int i = 0; i < currentLine.numberOfObjects; ++i) 
		{
            	    ObjectFile obj;

		    std::getline(stream, element, ',');
		    obj.label = std::stoi(element);
		    std::getline(stream, element, ',');
		    obj.id = std::stoi(element);
		    std::getline(stream, element, ',');
		    obj.posX = std::stoi(element);
		    std::getline(stream, element, ',');
		    obj.posY = std::stoi(element);
		    std::getline(stream, element, ',');
		    obj.velX = std::stof(element);
		    std::getline(stream, element, ',');
		    obj.velY = std::stof(element);
		    std::getline(stream, element, ',');
		    obj.width = std::stoi(element);
		    std::getline(stream, element, ',');
		    obj.height = std::stoi(element);

		    currentLine.objects.push_back(obj);
		}
        	parsedData.push_back(currentLine);
    	}
    	
    	file.close();
    	return parsedData;
} 


int main(int argc, char** argv)
{
	if (argc < 2) 
	{
		std::cerr << "Usage: " << argv[0] << " <inputFile> <show_screen>" << std::endl;
		std::cerr << "    <inputFile> must be a path to the input file" << std::endl;
		std::cerr << "    <show_screen> with value 1=show scenario, 0=don't show" << std::endl; 
		return 1;
	}

	std::vector<std::string> data_to_csv;
	std::string outputFileName(std::string(argv[1])+".txt");
	TrackerParam params;
	params.read(std::string("config/params.txt"));	
	std::string input_folder = "data/"+std::string(argv[1]);
	Tracker tracker(params);
	std::vector<Detection> dets;
	
  	int img_width, img_height;
	cv::Mat image;
	cv::Rect rect;
	Line curr;
	std::vector<Line> detections = readerGT(input_folder + "/out.txt");

	std::ifstream file;
	std::string filename = input_folder + "/setup.txt";
	file.open(filename); 
	std::string line;
	while(std::getline(file, line))
	{
		if(line.empty())
			continue;
		else if(line.find("Image width") != std::string::npos)
		{
			std::string el = line.substr(line.find(":")+1);
			img_width = atoi(el.c_str()); 
		}
		else if(line.find("Image height") != std::string::npos)
		{
			std::string el = line.substr(line.find(":")+1);
			img_height = atoi(el.c_str());
		}
		else
			continue;
	}
	

	std::vector<std::chrono::microseconds::rep> global_times, track_times;
	std::vector<std::chrono::microseconds::rep> lkf_predict_times, lkf_update_times;
	std::vector<std::chrono::microseconds::rep> gnn_times, trackUpdate_times;
	std::vector<double> rmse_vec;
	int total_frame_num = detections.size();
	int total_num_objects = 0;
	int frame_num_withObj = 0;	

	for(uint i=0; i < detections.size(); ++i)
	{
		dets.clear();
		curr = detections[i];
		image = cv::Mat(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));
		int j = 0;
		std::stringstream ss;

		for(auto obj : curr.objects)
		{				
			Detection d(Eigen::Vector3f(obj.posX, obj.posY,0), Eigen::Vector3f(obj.width, obj.height, 0), obj.label);
			d.setVelocity(Eigen::Vector3f(obj.velX, obj.velY,0));
			dets.push_back(d);

     	 	if (SHOW_SCREEN)
			{
				rect = cv::Rect(obj.posX, obj.posY, obj.width, obj.height);
				cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1 );
				ss.str("");
				ss << j;
				cv::putText(image, ss.str(), cv::Point(obj.posX,obj.posY), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			}
      	j++;
		}

		auto start_global_time = std::chrono::high_resolution_clock::now();
		tracker.track(dets);
		auto end_global_time = std::chrono::high_resolution_clock::now(); 
		auto duration_global = std::chrono::duration_cast<std::chrono::microseconds>(end_global_time - start_global_time).count();

		if (dets.size() != 0 && i>0)
		{
			frame_num_withObj += 1;
			rmse_vec.push_back(tracker.getRMSE());
			global_times.push_back(duration_global);  
			track_times.push_back(tracking_time);	
			trackUpdate_times.push_back(trackUpd_time);
			lkf_predict_times.push_back(lkf_predict_time);
			lkf_update_times.push_back(lkf_update_time);
			gnn_times.push_back(gnn_time);
		}


    	if (SHOW_SCREEN)
		{
			cv::Mat trackingImg = image.clone();
			tracker.drawTracks(trackingImg, img_width, img_height);
			std::string img_path = "data/results/res" + std::to_string(i) + ".jpg";
			cv::Mat m = mosaic(image, trackingImg);
			cv::imwrite(img_path, trackingImg);
			// cv::imshow("DataTracker", image);					
			// cv::waitKey(cvRound(millisec));
		}

		if(DUMP_ON)
		{
			std::vector<std::vector<int>> currentTracks = tracker.getTracks();
			for (auto track : currentTracks)
			{
				data_to_csv.push_back(std::to_string(i+1)+","); 			//frame
				data_to_csv.push_back(std::to_string(track[0])+",");		//id
				data_to_csv.push_back(std::to_string(track[1])+",");		//bb_left (top-left corner o è il centro?)
				data_to_csv.push_back(std::to_string(track[2])+",");		//bb_top  (top-left corner o è il centro?)
				data_to_csv.push_back(std::to_string(track[3])+",");		//bb_width
				data_to_csv.push_back(std::to_string(track[4])+",");		//bb_height
				data_to_csv.push_back("1,");
				data_to_csv.push_back("-1,");
				data_to_csv.push_back("-1,");
				data_to_csv.push_back("-1\n");
			}
		}

		dets.clear();
	}

	//write output
	write_file(data_to_csv, outputFileName);

	// Compute sum of values
	float rmse_sum = std::accumulate( rmse_vec.begin(), rmse_vec.end(), 0.0) ;
	float global_times_sum = std::accumulate( global_times.begin(), global_times.end(), 0.0);
	float track_times_sum = std::accumulate( track_times.begin(), track_times.end(), 0.0);
	float tracks_update_sum = std::accumulate( trackUpdate_times.begin(), trackUpdate_times.end(), 0.0);
	float predict_times_sum = std::accumulate( lkf_predict_times.begin(), lkf_predict_times.end(), 0.0);
	float update_times_sum = std::accumulate( lkf_update_times.begin(), lkf_update_times.end(), 0.0);	
	float association_times_sum = std::accumulate( gnn_times.begin(), gnn_times.end(), 0.0);

	//compute values per frame (pf)
	float obj_pf = (float) total_num_objects / frame_num_withObj;
	float rmse_pf = (float) rmse_sum / total_frame_num;
	float global_time_pf = (float) global_times_sum / frame_num_withObj;
	float tracking_time_pf = (float) track_times_sum / frame_num_withObj;
	float track_update_time_pf = (float) tracks_update_sum / frame_num_withObj;
	float predict_time_pf = (float) predict_times_sum / frame_num_withObj;
	float update_time_pf = (float) update_times_sum / frame_num_withObj;
	float association_time_pf = (float) association_times_sum / frame_num_withObj;

	std::cout<<"\n@@@@@@@@@@@@@@@@@@@\n";
	std::cout<<"predict time pf: "<<predict_time_pf<<" microsec\n";
	std::cout<<"association time pf: "<<association_time_pf<<" microsec\n";
	std::cout<<"create tracks time pf: "<<track_update_time_pf<<" microsec\n";
	std::cout<<"update time pf: "<<update_time_pf<<" microsec\n";
	std::cout<<"GLOBAL time pf: "<<global_time_pf<<" microsec\n";
	std::cout<<"GLOBAL RMSE pf: "<<rmse_pf<<" pixels\n";	//quanti metri per ogni pixel?
	std::cout<<"@@@@@@@@@@@@@@@@@@@\n";
	return 0;
}
