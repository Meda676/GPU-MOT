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