
#pragma once


#include "../../differentials/roots.hpp"
#include "../../matrix/matrix.hpp"
#include "../../matrix/matrix_map.hpp"

/*#####################################################################
 Potential is diagonal matrix, therfore entries are stored in a vector
 ######################################################################*/

namespace qwv{
 namespace potential{
 
 template<typename T>
 auto constexpr gaussian2D(std::size_t nR, std::size_t nr, double LR, double Lr, double V12, double V13, double V23){
   
      double val_a, val_b, val_c;
      std::vector<T>  x_TBCheb_R(nR), x_TBCheb_r(nr), GaussPotential; GaussPotential.reserve(nR*nr);
     
      auto v_nR = std::views::iota(static_cast<std::size_t>(0), nR);
      auto v_nr = std::views::iota(static_cast<std::size_t>(0), nr);

      auto rootsR = qwv::discretization::roots2D<T>(nR);
      auto rootsr = qwv::discretization::roots2D<T>(nr);


       std::for_each(std::execution::par_unseq, std::begin(v_nR), std::end(v_nR), [&](auto i){
         x_TBCheb_R[i] = LR*rootsR[i]/sqrt(1-pow(rootsR[i],2));
       });


       std::for_each(std::execution::par_unseq, std::begin(v_nr), std::end(v_nr), [&](auto i){
         x_TBCheb_r[i] = Lr*rootsr[i]/std::sqrt(1-std::pow(rootsr[i],2));
       });

       std::for_each(std::execution::par_unseq, std::begin(v_nR), std::end(v_nR), [&](auto i){
         for(auto j : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(nr)){
            val_a = std::pow(x_TBCheb_R[i],2);
            val_b = 0.25*std::pow(x_TBCheb_R[i],2)+std::pow(x_TBCheb_r[j],2)+x_TBCheb_R[i]*x_TBCheb_r[j];
            val_c = 0.25*std::pow(x_TBCheb_R[i],2)+std::pow(x_TBCheb_r[j],2)-x_TBCheb_R[i]*x_TBCheb_r[j];
            GaussPotential.push_back((-(V12*std::exp(-val_a)+V13*std::exp(-val_b)+V23*std::exp(-val_c))));
        }
       });

     return GaussPotential;
     }
 
 
 template<typename T>
 constexpr auto gaussian4D(std::size_t nR, std::size_t nr, double LR, double Lr, double V12, double V13, double V23){

    std::vector<double> result; result.reserve(nR*nR*nr*nr);
    std::vector<double> x_TBCheb_R(nR), x_TBCheb_r(nr);
     
     double val_a, val_b, val_c, val_B, temp, temp2, temp3;
     
     auto v_nR = std::views::iota(static_cast<std::size_t>(0), nR);
     auto v_nr = std::views::iota(static_cast<std::size_t>(0), nr);
     
     auto rootsR = qwv::discretization::roots<T>(nR);
     auto rootsr = qwv::discretization::roots<T>(nr);

     std::for_each(std::execution::par_unseq, std::begin(v_nR), std::end(v_nR), [&](auto i){
       x_TBCheb_R[i] = LR*rootsR[i]/sqrt(1-pow(rootsR[i],2));
     });


     std::for_each(std::execution::par_unseq, std::begin(v_nr), std::end(v_nr), [&](auto i){
       x_TBCheb_r[i] = Lr*rootsr[i]/std::sqrt(1-std::pow(rootsr[i],2));
     });

  
         for(auto i : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(nR) ){
            for(auto j : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(nR)){
                    for(auto k : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(nr)){
                            for(auto l : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(nr)){
              val_a = std::pow(x_TBCheb_R[i],2) + std::pow(x_TBCheb_R[j],2);
              temp  = std::pow(x_TBCheb_r[k],2) +std::pow(x_TBCheb_r[l],2);
              temp2= x_TBCheb_R[i]*x_TBCheb_r[k]+x_TBCheb_R[j]*x_TBCheb_r[l];

              val_b = 0.25*val_a+ temp + temp2;
              val_c = 0.25*val_a+ temp - (temp2);

              result.push_back( -(V12*std::exp(-val_a) + V13*std::exp(-val_b) + V23*std::exp(-val_c) ));
                            }
                    }
              }
         }

     return result;
 }


 } // end of namespace potential
} // end of namespace qwv
