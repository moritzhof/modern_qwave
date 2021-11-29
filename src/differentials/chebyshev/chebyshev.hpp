
#pragma once
#include <execution>

#include "../roots.hpp"
#include "../../matrix/matrix.hpp"
#include "../../matrix/matrix_map.hpp"

namespace qwv{
 namespace differential{

   template<typename T>
   auto constexpr Chebyshev1D(auto&& roots, std::size_t N){

       
       qwv::matrix<double> Chebyshev(N, N);
       for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
         (i == j) ?
         Chebyshev(i,j) = 0.5*roots[i]/(1.0-(roots[i]*roots[i]))
         :
         Chebyshev(i,j) = std::cos(M_PI*(i+j))*std::sqrt((1.0-(roots[j]*roots[j]))/(1.0-(roots[i]*roots[i])))/(roots[i]-roots[j]);
         }
       }
       
     roots.free();
     return Chebyshev;
   }
 
   template<typename T>
   auto constexpr Chebyshev2D(auto&& roots, std::size_t N){

     auto Chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);

       qwv::matrix<T> Chebyshev2D(N,N);
       for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
             (i == j) ?
             Chebyshev2D(i,j) = pow(roots[i],2)/pow(1.0-pow(roots[i],2),2)-(pow(N,2)-1)/(3.0*(1.0-pow(roots[i],2)))
             :
             Chebyshev2D(i,j) = Chebyshev1D(i,j)*(roots[i]/(1.0-pow(roots[i],2))-2.0/(roots[i]-roots[j]));
           }
       }
     
     return Chebyshev2D;
   }
  
 
   template<typename T>
   auto constexpr Chebyshev1DTB(auto&& roots, std::size_t N, double L){
     
       std::vector<T> A11; A11.reserve(N);
       qwv::matrix<T> chebyshev1DTB(N,N);
       
       auto Chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);
     
       for(auto i : std::views::iota(std::size_t(0)) | std::views::take(N) ){
         A11[i] = 1.0/L*std::pow(std::sqrt(1.0-std::pow(roots[i],2)),3);
       }
       
       for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
             chebyshev1DTB(i,j) = A11[i]*Chebyshev1D(i,j);
         }
       }
       return chebyshev1DTB;
 }
 
 template<typename T>
 auto constexpr Chebyshev2DTB(auto&& roots, std::size_t N, double L){

     std::vector<T> A21, A22; A21.reserve(N); A22.reserve(N);
     qwv::matrix<T> chebyshev2DTB(N,N);
     qwv::matrix<T> temp(N,N);
    
     auto Chebyshev1D = qwv::differential::Chebyshev1D<double>(roots, N);
     auto Chebyshev2D = qwv::differential::Chebyshev2D<double>(roots, N);

     for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         A21[i]= -3.0/std::pow(L,2)*roots[i]*std::pow(1.0-std::pow(roots[i],2),2);
         A22[i]=  1.0/std::pow(L,2)*std::pow(1.0-std::pow(roots[i],2),3);
       }

     for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
      for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
           chebyshev2DTB(i,j) = A22[i]*Chebyshev2D(i,j);
           temp(i,j)          = A21[i]*Chebyshev1D(i,j);
           chebyshev2DTB(i,j) = temp(i,j) + chebyshev2DTB(i,j);
        }
      }
     return chebyshev2DTB;

}
 
 } //end of discretization namespace
} // end of qwv namespace
