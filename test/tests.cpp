
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
   
    double tol = 1.0e-9;  //std::numeric_limits<double>::epsilon();
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0){
        
       
//###################################################################
//#################### Chebyshev 1D #################################
    {
        std::size_t N = 10;
        double L = 0.5;
        auto roots = qwv::discretization::roots<double>{N};
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
    // TO DO
        
    } // end of scope N = 10
        
        
{ //######################## SCOPE N = 25 ######################
    std::size_t N = 25;
    double L = 0.5;
    auto roots = qwv::discretization::roots<double>{N};
    
{
std::string filename = "input_files/chebyshev1D_25.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1D, tol );
}
    
{
std::string filename = "input_files/chebyshev2D_25.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev2D = qwv::differential::Chebyshev2D<double>(roots, N);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2D, tol );
}
    
//###################################################################
//#################### Chebyshev 1TBD ###############################
{
std::string filename = "input_files/chebyshev1DTB_25.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev1DTB = qwv::differential::Chebyshev1DTB<double>(roots, N, L);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1DTB, tol );
}
//###################################################################
//#################### Chebyshev 2TBD ###############################
{
std::string filename = "input_files/chebyshev2DTB_25.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev2DTB = qwv::differential::Chebyshev2DTB<double>(roots, N, L);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2DTB, tol );
}
            
    
}//end of scope N = 25
        
{ //######################## SCOPE N = 100 ######################
    std::size_t N = 100;
    double L = 0.5;
    auto roots = qwv::discretization::roots<double>{N};
{
std::string filename = "input_files/chebyshev1D_100.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1D, tol );
}
    
//###################################################################
//#################### Chebyshev 2D #################################
{
std::string filename = "input_files/chebyshev2D_100.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev2D = qwv::differential::Chebyshev2D<double>(roots, N);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2D, tol );
}
    
//###################################################################
//#################### Chebyshev 1TBD ###############################
{
std::string filename = "input_files/chebyshev1DTB_100.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev1DTB = qwv::differential::Chebyshev1DTB<double>(roots, N, L);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev1DTB, tol );
}
//###################################################################
//#################### Chebyshev 2TBD ###############################
{
std::string filename = "input_files/chebyshev2DTB_100.txt";
auto matlab_chebyshev = read_file(filename, N, N);
auto chebyshev2DTB = qwv::differential::Chebyshev2DTB<double>(roots, N, L);
qwv::test::compare<double>(N, matlab_chebyshev, chebyshev2DTB, tol );
}
                    
} // end of scope N = 100
        
        
        
}//end of rank == 0;
    
    MPI_Finalize();
    return 0;
}
