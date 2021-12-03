#ifdef __CUDACC__

#include "cutlass/gemm/device/gemm.h"
#include <cublas_v2.h>
/*########################################################
      These implementations assuming column-major order
 ########################################################*/

namespace qwv{
 namespace cuda{
  
 __global__
 void dgemm1(int M, int N, int K, double alpha, double const *A, double const *B, double beta, double *C) {
     
     typedef block_task_policy < 128, 32, 8, 4,8, true, block_raster_enum::Default> block_task_policy_t;
     
     typedef gemm::blas_scaled_epilogue<double, double, double> gemm_op_t;
     
     typedef block_task< block_task_policy_t, double, double, matrix_transform_t::Transpose, 4, matrix_transform_t::NonTranspose, 4, gemm_op_t, 4, true > block_task_t;

     __shared__ block_task_t::scratch_storage_t smem;

     block_task_t(reinterpret_cast(&smem), &smem, A, B, C, gemm_op_t(alpha, beta), M, N, K).run();
}
 
 
 __global__
 void dgemm2(int M, int N, int K, double alpha, double const *A, int lda,
  double const *B, int ldb, double beta, double *C, int ldc) {

     typedef block_task_policy < 128, 32, 8, 4,8, true, block_raster_enum::Default> block_task_policy_t;
     
     typedef gemm::blas_scaled_epilogue<double, double, double> gemm_op_t;
     
     typedef block_task< block_task_policy_t,double,double,matrix_transform_t::NonTranspose,4, matrix_transform_t::NonTranspose, 4, gemm_op_t, 4, true > block_task_t ;

     __shared__ block_task_t::scratch_storage_t smem;

     block_task_t(reinterpret_cast(&smem), &smem, A, B, C, gemm_op_t(alpha, beta), M, N, K).run();

 } // end of cuda namespace
} // end of qwv namespace

namespace qwv{
 namespace cublas{
 void dgemm1(const qwv::cuda::device_ptr<double> A, const qwv::cuda::device_ptr<double> B, qwv::cuda::device_ptr<T> C,
            const int m, const int k, const int n) {
       int lda=m,ldb=k,ldc=m;
       const double alf = 1;
       const double bet = 0;
       const double *alpha = &alf;
       const double *beta = &bet;
 

      cublasHandle_t handle;
      cublasCreate(&handle);
 
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
     cublasDestroy(handle);
    }
 
 
 void dgemm2(const qwv::cuda::device_ptr<double> A, const qwv::cuda::device_ptr<double> B, qwv::cuda::device_ptr<T> C,
            const int m, const int k, const int n) {
     
       int lda=m,ldb=k,ldc=m;
       const double alf = 1;
       const double bet = 0;
       const double *alpha = &alf;
       const double *beta = &bet;
 

      cublasHandle_t handle;
      cublasCreate(&handle);
 
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
     cublasDestroy(handle);
    }
 } // end of cuda namespace
} // end of qwv namespace
#endif
