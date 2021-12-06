
#include <vector>
#include "../differentials/chebyshev/chebyshev.hpp"
#include "operator_tensor.hpp"

 
  __global__ void buildOperator2D(qwv::cuda::device_ptr<double> rootsR, qwv::cuda::device_ptr<double> rootsr, std::size_t nR, std::size_t nr, double LR, double Lr, int parity1, int parity2, double a1, double a2, std::vector<double> const& V){

      qwv::discretization::cuda::roots<double>(rootsR, nR);
      qwv::discretization::cuda::roots<double>(rootsr, nr);
            
      qwv::differential::cuda::Chebyshev1D<double>(chebyshev1DR, nR);
      qwv::differential::cuda::Chebyshev2D<double>(chebyshev2DR,chevbyshev1DR, nR);
      __syncthreads()
      
      qwv::differential::cuda::Chebyshev1D<double>(chevbyshev1Dr, nr);
      qwv::differential::cuda::Chebyshev2D<double>(chebyshev2Dr,chevbyshev1Dr, nr);
      __syncthreads()

      qwv::differential::cuda::Chebyshev2DTB<double>(rootsr, nr, Lr);
      qwv::differential::cuda::Chebyshev2DTB<double>(rootsr, nr, Lr);

      double* TBChebD2R11 = (double*)malloc(nR*nR*sizeof(T));
      double* TBChebD2R12 = (double*)malloc(nR*nR*sizeof(T));
      double* TBChebD_2R  = (double*)malloc(nR*nR*sizeof(T));
      double* TBChebD2r11 = (double*)malloc(nr*nr*sizeof(T));
      double* TBChebD2r12 = (double*)malloc(nr*nr*sizeof(T));
      double* TBChebD_2r(nr, nr) = (double*)malloc(nr*nr*sizeof(T));


     #pragma unroll
      for(int i = 0; i < nR; ++i){
         for(auto j=0; j<nR; j++){
            TBChebD2R11[i*nR+j]=cheby2DTBR[i*2*nR+j];
            TBChebD2R12[i*nR+j]=parity1*cheby2DTBR[i*2*nR+2*nR-1-j];
         }
      }

    #pragma unroll
    for(int i = 0; i < nr; ++i){
        for(auto j=0; j<nr; j++){
            TBChebD2r11[i*nr+j]=cheby2DTBr[i*2*nr+j];
            TBChebD2r12[i*nr+j]=parity2*cheby2DTBr[i*2*nr+2*nr-1-j];
        }
    }

    #pragma unroll
    for(int i = 0; i < nr; ++i){
        for(auto j=0; j<nr; j++){
             TBChebD_2r[i*nr+j] = TBChebD2r11[i*nr+j] + TBChebD2r12[i*nr+j];
        }
    }

    #pragma unroll
    for(int i = 0; i < nR; ++i){
     for(auto j=0; j<nR; j++){
        TBChebD_2R[i*nr+j] = TBChebD2R11[i*nr+j] + TBChebD2R12[i*nr+j];
      }
    }
    //Convert to operator format

    // Contribution 1: a1 * (TBChebD2R (x) I_nr)
    // Contribution 2: a2 * (I_nR (x) TBChebD2r)
    // Contribution 3: Vsp
    // note: the order is exchanged in our operator definition as compared to the original sparse matrix construction
    //std::unique_ptr<qwv::kron2D<double>> kron2D(new qwv::kron2D<double>(nR, TBChebD_2r, nr, TBChebD_2R, V, a2, a1));
   // return kron2D;
  }
