
#include <iomanip>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "../src/matrix/matrix_map.hpp"
#include "../src/matrix/matrix.hpp"
#include "../src/differentials/chebyshev/chebyshev.hpp"

#include "file_reader.hpp"
#include <mpi.h>

auto main(int argc, char* argv[])->int{
    int rank;
    double L = 0.5;
    std::size_t N = 10;
    auto roots = qwv::discretization::roots<double>{N};
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0){
        
       
        double tol = 1.0e-12;  //std::numeric_limits<double>::epsilon();
//###################################################################
//#################### Chebyshev 1D #################################
    {
    std::string filename = "input_files/chebyshev1D.txt";
    auto matlab_chebyshev = read_file(filename, N, N);
    auto chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);
    qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1D, tol );
    }
        
// ..... TEST TO FAIL ....
//    {
//    std::string filename = "input_files/chebyshev1D.txt";
//    auto matlab_chebyshev = read_file(filename, N, N);
//    qwv::matrix<double> fail_chebyshev1D(N,N);
//    compare<double>(N, matlab_chebyshev, fail_chebyshev1D, tol );
//    }
//###################################################################
//#################### Chebyshev 2D #################################
    {
    std::string filename = "input_files/chebyshev2D.txt";
    auto matlab_chebyshev = read_file(filename, N, N);
    auto chebyshev2D = qwv::differential::Chebyshev2D<double>(roots, N);
    qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2D, tol );
    }
        
//###################################################################
//#################### Chebyshev 1TBD ###############################
    {
    std::string filename = "input_files/chebyshev1DTB.txt";
    auto matlab_chebyshev = read_file(filename, N, N);
    auto chebyshev1DTB = qwv::differential::Chebyshev1DTB<double>(roots, N, L);
    qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1DTB, tol );
    }
//###################################################################
//#################### Chebyshev 2TBD ###############################
    {
    std::string filename = "input_files/chebyshev2DTB.txt";
    auto matlab_chebyshev = read_file(filename, N, N);
    auto chebyshev2DTB = qwv::differential::Chebyshev2DTB<double>(roots, N, L);
    qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2DTB, tol );
    }
//###################################################################
//################### Derivative test ###############################
//    {
//        auto grid = even_grid(N, -1, 1);
//        auto cos = std::vector<double>{};
//        for(auto const& i : grid){
//            cos.push_back(std::cos(i));
//        }
//        
//        auto solution = std::vector<double>{};
//        for(int i = 0; i < N; ++i){
//            solution.push_back(std::cos(roots[i]));
//        }
//        for(auto const& i : solution){ std::cout << i << ' ';
//        }std::cout << '\n';
// 
//        auto u = std::vector<double>{};
//        for(int i = 0; i < N; ++i){
//            u.push_back(std::sin(roots[i]));
//        }
//     
//        auto chebyshev1D = qwv::differential::Chebyshev1DTB<double>(roots, N, 0.44); // or just 1D
//        auto result = std::vector<double>(N);
//
//        for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
//          for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
//              result[i] += chebyshev1D[i*N+j]*u[i];
//          }
//        }
//
//        for(auto const& i : result){ std::cout << i << ' ';
//        }std::cout << '\n';
//        qwv::test::compare_vector<double>(N, solution, result, tol );
//
//    }
}//end of rank == 0;
    
    MPI_Finalize();
    return 0;
}
