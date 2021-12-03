#pragma once

#include <vector>
#include "../differentials/chebyshev/chebyshev.hpp"
#include "operator_tensor.hpp"


using dense_matrix = qwv::matrix<double>;

  // Construct an operator representation of the three-body problem.
  // Interface equivalent to buildTotalSparseMatrix2D
  std::unique_ptr<qwv::kron2D<double>> buildOperator2D(std::size_t nR, std::size_t nr, double LR, double Lr, int parity1, int parity2, double a1, double a2, std::vector<double> const& V){

         auto rootsR = qwv::discretization::roots<double>{nR};
         auto rootsr = qwv::discretization::roots<double>{nr};
      
          // build differential matrices
         auto cheby2DTBR = qwv::differential::Chebyshev1DTB<double>(rootsR, nR, LR);
         auto cheby2DTBr = qwv::differential::Chebyshev2DTB<double>(rootsr, nr, Lr);

          dense_matrix TBChebD2R11(nR, nR), TBChebD2R12(nR, nR), TBChebD_2R(nR, nR);
          dense_matrix TBChebD2r11(nr, nr), TBChebD2r12(nr, nr), TBChebD_2r(nr, nr);


            auto v_nR = std::views::iota(static_cast<std::size_t>(0), nR);
            std::for_each(std::execution::par, std::begin(v_nR), std::end(v_nR), [&](auto i){
                 for(auto j=0; j<nR; j++){
                         TBChebD2R11[i*nR+j]=cheby2DTBR[i*2*nR+j];
                         TBChebD2R12[i*nR+j]=parity1*cheby2DTBR[i*2*nR+2*nR-1-j];
                }
            });

            auto v_nr = std::views::iota(static_cast<std::size_t>(0), nr);
            std::for_each(std::execution::par, std::begin(v_nr), std::end(v_nr), [&](auto i){
                 for(auto j=0;j<nr;j++){
                         TBChebD2r11[i*nr+j]=cheby2DTBr[i*2*nr+j];
                         TBChebD2r12[i*nr+j]=parity2*cheby2DTBr[i*2*nr+2*nr-1-j];
                }
            });

            //Total differential matrix
             TBChebD_2R = TBChebD2R11 + TBChebD2R12;
             TBChebD_2r = TBChebD2r11 + TBChebD2r12;


    //Convert to operator format

    // Contribution 1: a1 * (TBChebD2R (x) I_nr)
    // Contribution 2: a2 * (I_nR (x) TBChebD2r)
    // Contribution 3: Vsp
    // note: the order is exchanged in our operator definition as compared to the original sparse matrix construction
    std::unique_ptr<qwv::kron2D<double>> kron2D(new qwv::kron2D<double>(nR, TBChebD_2r, nr, TBChebD_2R, V, a2, a1));
    return kron2D;
  }

// Construct an operator representation of the 2D Laplacian on an n x n Chebyshev mesh covering an L x L domain
std::unique_ptr<qwv::kron2D<double>> buildLaplaceOperator2D(std::size_t n, double L){
    
    auto roots = qwv::discretization::roots<double>{n};
    // build differential matrices
    auto TBChebD2 = qwv::differential::Chebyshev2DTB<double>(roots, n, L);
    std::vector<double> Vdum;
    std::unique_ptr<qwv::kron2D<double>> kron2D(new qwv::kron2D<double>(n, TBChebD2,n,TBChebD2,Vdum,1.0,1.0));
    return kron2D;
}

#ifdef HAVE_PHIST
std::unique_ptr<qwv::kron4D<double>> buildOperator4D(std::size_t nR, std::size_t nr,  double LR, double Lr, int parity1, int parity2, double a1, double a2, std::vector<double> const& V){

    std::unique_ptr<qwv::kron2D<double>> kron2d_R = buildLaplaceOperator2D(nR, LR);
    std::unique_ptr<qwv::kron2D<double>> kron2d_r = buildLaplaceOperator2D(nr, Lr);
    std::unique_ptr<qwv::kron4D<double>> kron4D(new qwv::kron4D<double>(*kron2d_r,*kron2d_R,V,a2,a1,MPI_COMM_WORLD));
return kron4D;

}
#endif



