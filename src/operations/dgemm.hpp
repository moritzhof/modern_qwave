#pragma once

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

//#ifdef PHIST_HAVE_MKL
#include "mkl_cblas.h"
//#else
//#include "cblas.h"
//#endif

namespace qwv{

    void dgemm1(int nrowsA, int ncolsA, int ncolsB, double const* RESTRICT A, int ldA,
                 double const* RESTRICT B, int ldB, double* RESTRICT C, int ldC, double alpha, double beta){
        
        cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans,
                 nrowsA, ncolsB, ncolsA,
                 alpha, A, ldA, B, ldB,
                 beta, C, ldC);
    }

//! standard double GEMM that computes C=alpha*A*Bt+beta*C.
//! A, Bt and C should be column-major
    void dgemm2(int nrowsA, int ncolsA, int ncolsBt, double const* RESTRICT A, int ldA,
                double const* RESTRICT Bt, int ldBt, double* RESTRICT C, int ldC, double alpha, double beta){
        
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 nrowsA, ncolsBt, ncolsA,
                 alpha, A, ldA, Bt, ldBt,
                 beta, C, ldC);
    }


}//end of namespace qwv
