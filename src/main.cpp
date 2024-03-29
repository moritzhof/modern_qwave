
#include <iostream>
#include "qwv_util.hpp"

#include "coord/coord.hpp"
#include "matrix/matrix_map.hpp"
#include "matrix/matrix.hpp"

#include "differentials/chebyshev/chebyshev.hpp"
//#include "differentials/chebyshev/parallel/chebyshev.hpp"

#include "potentials/gaussian/gaussian.hpp"

#include "operators/operator_kronecker.hpp"
#include "operations/dgemm.hpp"

#include "preconditioner/sylvester.hpp"
#include "eigensolver/eigensolver.hpp"
#include "cuda/memory.cu"

auto main(int argc, char* argv[])->int{
    
    
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    std::size_t N = 10;
    double L = 0.5;
    if(rank == 0){
   
    auto roots = qwv::discretization::roots<double>{N};
        
    auto cheby1D = qwv::differential::Chebyshev1D<double>(roots, N);
    auto cheby2D = qwv::differential::Chebyshev2D<double>(roots, N);
    auto cheby1DTB = qwv::differential::Chebyshev1DTB<double>(roots, N, L);
    auto cheby2DTB = qwv::differential::Chebyshev2DTB<double>(roots, N, L);
        
        
    auto gaussian = qwv::potential::gaussian2D<double>(N, N, L, L, 0.5, 0.5, 0.5);
  //  auto gaussian2 = qwv::potential::gaussian4D<double>(N, N, L, L, 0.5, 0.5, 0.5);
        
        
    std::cout << cheby1D << '\n';
    std::cout << "###########################\n";
    std::cout << cheby2D << std::endl;
    std::cout<<'\n';
    std::cout << "###########################\n";
    std::cout << cheby1DTB << std::endl;
    std::cout << "###########################\n";
    std::cout << cheby2DTB << std::endl;
    std::cout << "###########################\n";
    qwv::util::print_vector(gaussian);
        
        std::cout << "###########################\n";
        qwv::matrix<double> mat(N, N);
        std::cout << mat << std::endl;
        
        
        
 /* CUDA PLAYGROUND
  
  
  double L = 0.5;
  std::size_t N = 10;
  
  auto roots = qwv::cuda::device_memory<double>::allocate_vector(N);
  auto cheby1D = qwv::cuda::device_memory<double>::allocate_matrix(N,N);
  auto cheby2D = qwv::cuda::device_memory<double>::allocate_matrix(N,N);
  auto cheby1DTB= qwv::cuda::device_memory<double>::allocate_matrix(N,N);
  auto cheby2DTB= qwv::cuda::device_memory<double>::allocate_matrix(N,N);
 
  dim3 blocks (2, 2, 1);
  dim3 threads(20, 20, 1);
  //qwv::roots<double> <<< 1, N>>>(roots, N);
  //Chebyshev1D<double><<<blocks,threads>>>(cheby1D, roots, N );
  Chebyshev2DTB<double><<<blocks,threads>>>(cheby2DTB, cheby2D, cheby1D, roots, N , L);
  qwv::cuda::synchronize();
  
  
  
  
  
  
  */
        
        
//    std::cout << "###########################\n";
//    qwv::util::print_vector(gaussian2);
        
//        qwv::matrix<double> mat(N, N);
//        std::cout << mat << std::endl;
//
//        qwv::matrix<int> mat1 = {{1,1,1}, {1,1,1}, {1,1,1}};
//
//        std::cout << mat1 << '\n';
//        qwv::matrix<int> mat2 = {{1,1,1}, {1,1,1}, {1,1,1}};
//        qwv::matrix<int> mat3 = mat1 + mat2;
//        std::cout << mat3 << '\n';
//        std::ranges::iota_view v{0,10};
//        for(auto i : v)
//        std::cout << i << std::endl;
    }

    MPI_Finalize();
    return 0;

}
