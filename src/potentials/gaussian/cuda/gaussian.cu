#ifdef __CUDACC__
#include<cuda.h>
#include "../../cuda/memory.cu"
#endif

//using qwv::cuda::device_ptr;

namespace qwv{
namespace potential{
 namespace cuda{
 
 template<typename T>
 __global__ void Gaussian2D(qwv::cuda::device_ptr<T> gaussian, qwv::cuda::device_ptr<T> rootsR, qwv::cuda::device_ptr<T> rootsr, std::size_t nR, std::size_t nr, double LR, double Lr, double V12, double V13, double V23){
     
     qwv::discretizaion::cuda::roots2D<T>(rootsR, nR);
     qwv::discretizaion::cuda::roots2D<T>(rootsr, nr);
     __syncthreads();
     
     __shared__ double val_a, val_b, val_c; double temp;
     __shared__  T* x_TBCheb_R = (T*)malloc(nR*sizeof(T));
     __shared__  T* x_TBCheb_r = (T*)malloc(nr*sizeof(T));
     
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     int j = threadIdx.y + blockIdx.y * blockDim.y;
     
     if(i < nR){
         x_TBCheb_R[i] = LR*rootsR[i]/sqrt(1-pow(rootsR[i],2));
     }
     __syncthreads();
     if(j < nr){
         x_TBCheb_r[j] = LR*rootsr[j]/sqrt(1-pow(rootsr[j],2));
     }
     __syncthreads();
     
    
     int index = 0;
     #pragma unroll
     for(int i = 0; i < nR; ++i){
         for(int j = 0; j < nr; ++j){
         val_a = pow(x_TBCheb_R[i],2);
         val_b = 0.25*pow(x_TBCheb_R[i],2)+pow(x_TBCheb_r[j],2)+x_TBCheb_R[i]*x_TBCheb_r[j];
         val_c = 0.25*pow(x_TBCheb_R[i],2)+pow(x_TBCheb_r[j],2)-x_TBCheb_R[i]*x_TBCheb_r[j];
             __syncthreads();
          temp = ((-(V12*exp(-val_a)+V13*exp(-val_b)+V23*exp(-val_c))));
          gaussian[index] = temp;
          index++;
         }
       }
   }
 
 
 
  } // end of cuda namespace
 }  // end of potential namespace
} // end of qwv namespace

