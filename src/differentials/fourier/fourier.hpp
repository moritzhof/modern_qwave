#pragma once


#include "../roots.hpp"
#include "../../matrix/matrix.hpp"
#include "../../matrix/matrix_map.hpp"

namespace qwv{
 namespace differential{

template<typename T>
auto constexpr FourierD1(auto&& fourier_roots, std::size_t N){


   qwv::matrix<T> Fourier(N,N);

    for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
      for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
       (i == j) ?
          Fourier(i,j) = 0
          :
          FourierD1(i,j) = 0.5*std::cos(M_PI*(i-j))/std::tan(0.5*(fourier_roots[i]-fourier_roots[j]));
    }
  }

  return Fourier;
}

template<typename T>
matrix<T> FourierD2(auto&& fourier_roots, std::size_t N){

  qwv::matrix<T> FourierD2(N,N);

    for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
      for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
      (i == j ?
       FourierD2(i,j) = -(1.0+0.5*std::pow(N,2))/6
       :
       FourierD2(i,j) = 0.5*std::cos(PI*(i-j+1))/(std::pow(std::sin(0.5*(x_Fourier[i]-x_Fourier[j])),2));
    }
  }
  return FourierD2;
}


 } //end of namespace differential
} //end of namespace qwv
