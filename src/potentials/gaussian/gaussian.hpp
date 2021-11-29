
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

       //matrix<T> GaussPotential(nR, nr);
       std::for_each(std::execution::par_unseq, std::begin(v_nR), std::end(v_nR), [&](auto i){
         for(int j = 0; j < nr; ++j){
            val_a = std::pow(x_TBCheb_R[i],2);
            val_b = 0.25*std::pow(x_TBCheb_R[i],2)+std::pow(x_TBCheb_r[j],2)+x_TBCheb_R[i]*x_TBCheb_r[j];
            val_c = 0.25*std::pow(x_TBCheb_R[i],2)+std::pow(x_TBCheb_r[j],2)-x_TBCheb_R[i]*x_TBCheb_r[j];
            GaussPotential.push_back((-(V12*std::exp(-val_a)+V13*std::exp(-val_b)+V23*std::exp(-val_c))));
        }
       });

     return GaussPotential;
     }

 } // end of namespace potential
} // end of namespace qwv
