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

#include "tracker_param.h"

void TrackerParam::read(const std::string& filename)
{
	std::ifstream file;
  
	try
  	{
		file.open(filename);
  	}
  	catch(...)
  	{
		std::cerr << "Cannot open " << filename << std::endl;
   	file.close();
		exit(-1);
  	}
  
  	if(!file.is_open())
  	{
   	std::cerr << "Error: file " << filename << " not found!" << std::endl;
   	exit(-1);
	}
  
	std::string line;
	while(std::getline(file, line))
	{
		std::remove_if(line.begin(), line.end(), isspace);
		if(line.empty())
		{
			continue;
		}
		else if(line.find("[PD]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				pd = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the PD: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[LOCAL_GSIGMA]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				g_sigma = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the LOCAL_GSIGMA: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[GLOBAL_GSIGMA]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				global_g_sigma = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the GLOBAL_GSIGMA: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[LOCAL_ASSOCIATION_COST]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				assocCost = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the LOCAL_ASSOCIATION_COST: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[GLOBAL_ASSOCIATION_COST]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				global_assocCost = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the GLOBAL_ASSOCIATION_COST: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[MAX_ASSOCIATION_COST]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				maxCost = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the MAX_ASSOCIATION_COST: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[LAMBDA]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				lambda = atof(line.c_str());
				gamma = lambda * 0.000001;
			}
			catch(...)
			{
				std::cerr << "Error in converting the LAMBDA: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[DELTA]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				std::istringstream iss(line);
				float a, b, c;
				if (!(iss >> a >> b >> c)) { break; }
				target_delta << a, b, c;
			}
			catch(...)
			{
				std::cerr << "Error in converting the DELTA: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[R_MATRIX]") != std::string::npos)
		{
			try
			{
				int row_size, col_size;
				std::getline(file, line);
				std::istringstream iss(line);
				iss >> row_size >> col_size;
				R = Eigen::MatrixXf(row_size, col_size);
				for (int i = 0; i < row_size; i++)
				{
					std::getline(file, line);
					std::istringstream iss(line);
					for (int j = 0; j < col_size; j++)
					{
						iss >> R(i,j);
					}
				}
			}
			catch(...)
			{
				std::cerr << "Error in converting the R VALUES: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[Q_MATRIX]") != std::string::npos)
		{
			try
			{
				int row_size, col_size;
				std::getline(file, line);
				std::istringstream iss(line);
				iss >> row_size >> col_size;
				Q = Eigen::MatrixXf(row_size, col_size);
				for (int i = 0; i < row_size; i++)
				{
					std::getline(file, line);
					std::istringstream iss(line);
					for (int j = 0; j < col_size; j++)
					{
						iss >> Q(i,j);
					}
				}
			}
			catch(...)
			{
				std::cerr << "Error in converting the Q VALUES: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[MAX_LIFE]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				maxLife = atoi(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the MAX LIFE value: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[MAX_NOT_DET]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				maxNotDet = atoi(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the MIN LIFE value: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[MIN_DET]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				minDet = atoi(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the MIN LIFE value: " << line << std::endl;
				exit(-1);
			}
		}
		else if(line.find("[DT]") != std::string::npos)
		{
			std::getline(file, line);
			try
			{
				dt = atof(line.c_str());
			}
			catch(...)
			{
				std::cerr << "Error in converting the DT: " << line << std::endl;
				exit(-1);
			}
		}
		else
		{
			std::cerr << "Option: " << line << " does not exist!" << std::endl;
			exit(-1);
		} 
	}
 
  file.close();
}
