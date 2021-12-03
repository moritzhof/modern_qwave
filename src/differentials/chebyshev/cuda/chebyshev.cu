
#include <cmath>
#include "../../roots.hpp"

#ifdef __CUDACC__
#include "../../../cuda/memory.cu"

using qwv::cuda::device_ptr;

namespace qwv{
  namespace differential{
     namespace cuda{
     
     template<typename T>
     __device__ void Chebyshev1D(device_ptr<T> Chebyshev Chebyshev, device_ptr<T> roots, std::size_t N){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i < N) && (j < N)){
    (i == j) ?
      Chebyshev[i*N+j] = 0.5*roots[i]/(1.0-(roots[i]*roots[i]))
      :
      Chebyshev[i*N+j] = cos(M_PI*(i+j))*sqrt((1.0-(roots[j]*roots[j]))/(1.0-(roots[i]*roots[i])))/(roots[i]-roots[j]);
    }
  }
//##################################################################################################
//##################################################################################################
     template<typename T>
     __device__ void Chebyshev2D(device_ptr<T> Chebyshev2D, device_ptr<T> Chebyshev1, device_ptr<T> roots, std::size_t N){
         
         qwv::differential::cuda::roots<T>(roots, N);
         qwv::differential::cuda::Chebyshev1D<T>(Chebyshev1, roots, N);

         int i = threadIdx.x + blockIdx.x * blockDim.x;
         int j = threadIdx.y + blockIdx.y * blockDim.y;
         
         if((i < N) && (j < N)){
               (i == j) ?
               Chebyshev2D[i*N+j]  = pow(roots[i],2)/pow(1.0-pow(roots[i],2),2)-(pow(N,2)-1)/(3.0*(1.0-pow(roots[i],2)))
               :
               Chebyshev2D[i*N+j]  = Chebyshev1[i*N+j]*(roots[i]/(1.0-pow(roots[i],2))-2.0/(roots[i]-roots[j]));
             }
         }
//##################################################################################################
//##################################################################################################
     
     
     template<typename T>
     __device__ void Chebyshev1DTB(device_ptr<T> chebyshev1DTB, device_ptr<T> Chebyshev1, device_ptr<T> roots, std::size_t N, double L){
       
         T* A11 = (T*)malloc(N*sizeof(T));

         qwv::differential::cuda::roots<T>(roots, N);
         qwv::differential::cuda::Chebyshev1D<T>(Chebyshev1, roots, N);

         int i = threadIdx.x + blockIdx.x * blockDim.x;
         int j = threadIdx.y + blockIdx.y * blockDim.y;

         if(i < N) { A11[i] = 1.0/L*std::pow(std::sqrt(1.0-std::pow(roots[i],2)),3); }
         
         if((i < N) && (j < N)){ chebyshev1DTB[i*N+j]  = A11[i]*Chebyshev1[i*N+j]; }
     
   }
     
//##################################################################################################
//##################################################################################################

   template<typename T>
   __device__ void Chebyshev2DTB(device_ptr<T> chebyshev2DTB, device_ptr<T> Chebyshev2, device_ptr<T> Chebyshev1, device_ptr<T> roots, std::size_t N, double L){

       T* A21 = (T*)malloc(N*sizeof(T));
       T* A22 = (T*)malloc(N*sizeof(T));
       T* temp = (T*)malloc(N*N*sizeof(T));
       
       qwv::differential::cuda::roots<T>(roots, N);
       qwv::differential::cuda::Chebyshev1D<T>(Chebyshev1, roots, N);
       qwv::differential::cuda::Chebyshev2D<T>(Chebyshev2, Chebyshev1, roots, N);
       
       
       int i = threadIdx.x + blockIdx.x * blockDim.x;
       int j = threadIdx.y + blockIdx.y * blockDim.y;

       if(i< N){
           A21[i]= -3.0/std::pow(L,2)*roots[i]*std::pow(1.0-std::pow(roots[i],2),2);
           A22[i]=  1.0/std::pow(L,2)*std::pow(1.0-std::pow(roots[i],2),3);
         }

       if((i < N) && (j < N)){
             chebyshev2DTB[i*N+j] = A22[i]*Chebyshev2[i*N+j];
             temp[i*N+j]          = A21[i]*Chebyshev1[i*N+j];
             __syncthreads();
             chebyshev2DTB[i*N+j] = temp[i*N+j] + chebyshev2DTB[i*N+j];
          }
   }

//##################################################################################################
//##################################################################################################
     
     
    } //end of cuda namespace
  } // end of differential namespace
} // end of qwv namespace
#endif
