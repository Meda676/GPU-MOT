/*** Basic Track class ***/ 

#include "track.h"

Track::Track(const float g_sigma, const float gamma, const int minDet, const int maxNotDet, 
				 const int maxLife) : g_sigma(g_sigma), gamma(gamma), minDetection(minDet), maxNotDetection(maxNotDet), maxLifeDuration(maxLife)
{
	increaseLifetime(1);
	entropy_sentinel = TrackState::ATTEMPT;
	attemptTime = 1;
	idT = -1;
}

void Track::trackUpdate()
{
	int currDet = getNumDet();
	if(entropy_sentinel == TrackState::ATTEMPT && currDet >= minDetection)
	{
		entropy_sentinel = TrackState::ACCEPT;
		attemptTime = 0;
	}
	else if (entropy_sentinel == TrackState::ATTEMPT && attemptTime >= maxLifeDuration)
	{
		entropy_sentinel = TrackState::DISCARD;
	}
	else if (serialMiss >= maxNotDetection)
	{
		entropy_sentinel = TrackState::DISCARD;
	}


}

bool Track::isAlive()
{ 
	if (entropy_sentinel == TrackState::DISCARD)
		return false;
	else
		return true; 
}
