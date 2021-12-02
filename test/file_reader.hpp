#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <ranges>
#include <algorithm>


#include "../src/matrix/matrix_map.hpp"
#include "../src/matrix/matrix.hpp"
#include "../src/differentials/chebyshev/chebyshev.hpp"

auto read_file(std::string& filename, std::size_t& nrow, std::size_t& ncol) {
  
    qwv::matrix<double> data(nrow, ncol);
    std::ifstream infile(filename, std::ios::in | std::ifstream::binary);
    
    for(int i = 0; i < nrow; ++i) {  // stop loops if nothing to read
       for(int j = 0; j < ncol; ++j){
            infile >> data[i*ncol+j];
        }
    }
    
    std::cout << "\033[1;33m done reading in MATLAB output file \033[0m: " << "\033[1;36m" << filename << "\033[0m\n";
  
  return data;
}

template<typename T>
void compare(std::size_t N, auto& compare1, auto& compare2, T tol = std::numeric_limits<T>::epsilon() ){
    
    auto result{0.0};

    auto i = std::views::iota(std::size_t(0), N*N);
    if( std::ranges::all_of(std::begin(i), std::end(i), [&](auto i){
        result = std::fabs(compare1[i]-compare2[i]);
        return result < tol; })){
            std::cout << "\033[1;32m TEST PASSED \033[0m\n";}
    else{
        std::cout << "\033[1;31m TEST FAILED \033[0m\n";
    }
}
