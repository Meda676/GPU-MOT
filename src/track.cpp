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
