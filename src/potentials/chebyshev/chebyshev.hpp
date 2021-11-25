
#pragma once

#include "../roots.hpp"

namespace qwv{
 namespace discretization{

   template<typename T>
   auto constexpr Chebyshev1D(std::size_t& N){

      auto roots = qwv::discretization::roots<double>(N);
      auto Chebyshev = std::vector<double>(N*N,0);

       for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
         (i == j) ?
           Chebyshev[i*N+i] = 0.5*roots[i]/(1.0-(roots[i]*roots[i]))
         :
           Chebyshev[i*N+j] = std::cos(M_PI*(i+j))*std::sqrt((1.0-(roots[j]*roots[j]))/(1.0-(roots[i]*roots[i])))/(roots[i]-roots[j]);
         }
       }
       
     roots.free();
     return Chebyshev;
   }
 
   template<typename T>
   auto constexpr Chebyshev2D(std::size_t& N){

     auto Chebyshev1D = qwv::discretization::Chebyshev1D<double>(N);
     auto roots = qwv::discretization::roots<double>(N);

     auto Chebyshev2D = std::vector<double>(N*N,0);
       for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
             (i == j) ?
             Chebyshev2D[i*N+j] = pow(roots[i],2)/pow(1.0-pow(roots[i],2),2)-(pow(N,2)-1)/(3.0*(1.0-pow(roots[i],2)))
             :
             Chebyshev2D[i*N+j] = Chebyshev1D[i*N+j]*(roots[i]/(1.0-pow(roots[i],2))-2.0/(roots[i]-roots[j]));
           }
       }

     return Chebyshev2D;
   }
  
 } //end of discretization namespace
} // end of qwv namespace
