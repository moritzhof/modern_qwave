#ifdef __CUDACC__
#include<cuda.h>
#endif

namespace qwv{
 namespace cuda{
 
     __global__
     void gaussian(std::size_t nR, std::size_t nr, double LR, double Lr, double V12, double V13, double V23){
         int i = threadIdx.x + blockIdx.x * blockDim.x;
           int j = threadIdx.y + blockIdx.y * blockDim.y;

         if (i < rows && j < columns) {
             
         }
     }
 
 
 
  } // end of cuda namespace
} // end of qwv namespace

