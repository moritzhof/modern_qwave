
#include <cmath>


#ifdef __CUDACC__
namespace qwv{
  namespace differential{
     namespace cuda{
     
     template<typename T>
__global__ void Chebyshev1D(T* Chebyshev, T* roots, std::size_t N){

    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i < N) && (j < N)){
    (i == j) ?
      Chebyshev[i*N+j] = 0.5*roots[i]/(1.0-(roots[i]*roots[i]))
      :
      Chebyshev[i*N+j] = cos(M_PI*(i+j))*sqrt((1.0-(roots[j]*roots[j]))/(1.0-(roots[i]*roots[i])))/(roots[i]-roots[j]);
    }
  }
//##################################################################################
     
    } //end of cuda namespace
  } // end of differential namespace
} // end of qwv namespace
#endif
