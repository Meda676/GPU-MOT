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
