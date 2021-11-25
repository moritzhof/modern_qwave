
#include <iostream>
#include "qwv_util.hpp"
#include "potentials/chebyshev/chebyshev.hpp"
#include "potentials/chebyshev/parallel/chebyshev.hpp"

#include "coord/coord.hpp"
#include "matrix/matrix_map.hpp"



auto main(int argc, char* argv[])->int{

std::size_t N = 10;
//auto cheby = util::to_vector(qwv::discretization::Chebyshev<double>(N));
auto cheby1D = qwv::discretization::Chebyshev1D<double>(N);
auto cheby2D = qwv::discretization::Chebyshev2D<double>(N);
    
auto par_cheby1D = qwv::discretization::parallel::Chebyshev1D<double>(N);

    for(const auto& i : cheby1D){
        std::cout<< i << ' ';
    }
    std::cout<<'\n';
    std::cout << " ###### COMPLETED: PARALLEL 1D ######## \n";
    for(const auto& i : par_cheby1D){
        std::cout<< i << ' ';
    }
    std::cout<<'\n';
    
    std::cout << " ###### COMPLETED: 1D ######## \n";
    for(const auto& i : cheby2D){
      std::cout<< i << ' ';
    }
    std::cout<<'\n';

    std::cout << " ###### COMPLETE ######## \n";
    
    return 0;

}
