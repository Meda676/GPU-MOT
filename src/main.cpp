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
#include <fstream>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>	
#include "tracker.h"
#include "detection.h"

bool SHOW_SCREEN;

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
	
	Tracker tracker;
	std::vector<Detection> dets;
	bool SHOW_SCREEN = atoi(argv[2]);

	Line curr;
	std::vector<Line> detections = readerGT(std::string(argv[1]));

  	cv::Mat image;
	const double millisec = 1000 / 7;
  	int img_width=1080;
	int img_height=1920;
	cv::Rect rect;
	
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

		tracker.track(dets);

    		if (SHOW_SCREEN)
		{
      			tracker.drawTracks(image, img_width, img_height);
			cv::imshow("DataTracker", image);					
			cv::waitKey(cvRound(millisec));	
		}

		dets.clear();
	}

	return 0;
}
