#pragma once

#include "../roots.hpp"
#include "../../matrix/matrix.hpp"
#include "../../matrix/matrix_map.hpp"

namespace qwv{
 namespace discretization{
   namespace parallel{


template<typename T>
   auto constexpr FourierD1(std::size_t N){

  auto v = std::views::iota(static_cast<std::size_t>(0), N);
  std::vector<T> x_Fourier(N); qwv::matrix<T> FourierD1(N,N);

  std::for_each(std::execution::par_unseq, std::begin(v), std::end(v),[&](auto i){
    x_Fourier[i] = ((2*i+1)*PI)/(N);
  });

  std::for_each(std::execution::par, std::begin(v), std::end(v),[&](auto i){
    for(auto j = 0; j < N; ++j){
      (i == j) ? FourierD1[i*N+j] = 0
      :
      FourierD1[i*N+j] = 0.5*std::cos(M_PI*(i-j))/std::tan(0.5*(x_Fourier[i]-x_Fourier[j]));
    }
  });

  return FourierD1;
}

template<typename T>
   auto constexpr FourierD2(std::size_t N){

  auto v = std::views::iota(static_cast<std::size_t>(0), N);
  std::vector<T> x_Fourier(N); qwv::matrix<T> FourierD2(N,N);

  std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
    x_Fourier[i] = ((2*i+1)*PI)/(N);
  });

    std::for_each(std::execution::par, std::begin(v), std::end(v),[&](auto i){
      for(auto j = 0; j < N; ++j){
      (i == j) ?
          FourierD2[i*N+j] = -(1.0+0.5*std::pow(N,2))/6;
          :
          FourierD2[i*N+j] = 0.5*std::cos(PI*(i-j+1))/(std::pow(std::sin(0.5*(x_Fourier[i]-x_Fourier[j])),2));
       }
    });
  return FourierD2;
}


   } //end of namespace parallel
  } //end of namespace discretization
} // end of namspace qwv
