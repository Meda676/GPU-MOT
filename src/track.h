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

#ifndef _TRACK_H_
#define _TRACK_H_

#include <iostream>
#include <Eigen/Core>
#include <numeric>

#include "kalman.h"
#include "lkf.h"

class Track
{
  public:
    Track() { ; }
    Track(const float g_sigma, const float gamma, const int minDet, const int maxNotDet, const int maxLife);

    void trackUpdate();
    bool isAlive();

    int getLifeTime() { return life_time.size(); }
    int getAttempt() { return attemptTime; }
    int getSerial() { return serialMiss; }
    int getNumDet() { return std::accumulate(life_time.begin(), life_time.end(), 0); }
    void resetNumDet() {life_time.clear();}
    
    enum class TrackState {ATTEMPT, ACCEPT, DISCARD};
    TrackState getEntropy() const { return entropy_sentinel; }
    void setIdTrack(const int& _id) { idT = _id; }
    int getIdTrack() const { return idT; }

    void increaseLifetime(int val) 
    {
      if (val == 1)
        serialMiss = 0;
      else 
        serialMiss++;

      if (entropy_sentinel == TrackState::ATTEMPT)
        attemptTime++;

      life_time.push_back(val); 

      if (life_time.size() > maxLifeDuration)
        life_time.erase(life_time.begin());
    }

  private:
    int idT;			   
    int serialMiss;		
    int attemptTime;
    int maxLifeDuration; 
    int maxNotDetection;  
    int minDetection;  
    float g_sigma;						  
    float gamma;	
    std::vector<int> life_time;
    TrackState entropy_sentinel;		
};

#endif
