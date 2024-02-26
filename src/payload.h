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

#ifndef _PAYLOAD_H_
#define _PAYLOAD_H_

struct matrixBuffer
{
   float *m;
   int m_r;
   int m_c;

   matrixBuffer() {}

   matrixBuffer(float *matrix, int cols, int rows)
   {
      m = matrix;
      m_r = rows;
      m_c = cols;
   }

   void reset()
   {
      m = nullptr;
   }

   void set(float* matrix_, int cols_, int rows_)
   {
      m = matrix_;
      m_c = cols_;
      m_r = rows_;
   }
};

typedef struct 
{
   matrixBuffer K;
   matrixBuffer F;
   matrixBuffer H;
   matrixBuffer G;
   matrixBuffer P;
   matrixBuffer S;
   matrixBuffer Sinv;
   matrixBuffer R;
   matrixBuffer Q;
   matrixBuffer x_filter;
   matrixBuffer P_predict;
   matrixBuffer x_predict;
   matrixBuffer z_predict;
   matrixBuffer z_measured;
   matrixBuffer bb_size;
   bool first;
} payload;

#endif