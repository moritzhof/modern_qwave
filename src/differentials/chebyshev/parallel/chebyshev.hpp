
#pragma once


namespace qwv{
 namespace discretization{
   namespace parallel{
 
   template<typename T>
   auto constexpr Chebyshev1D(auto&& roots, std::size_t N){

   auto v = std::views::iota(std::size_t(0), N);
   auto Chebyshev = std::vector<double>(N*N,0);

   std::for_each(std::execution::par_unseq, std::begin(v), std::end(v), [&](auto i){
     for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
     (i == j) ?
        Chebyshev[i*N+i] = 0.5*roots[i]/(1.0-(roots[i]*roots[i]))
      :
        Chebyshev[i*N+j] = std::cos(M_PI*(i+j))*std::sqrt((1.0-(roots[j]*roots[j]))/(1.0-(roots[i]*roots[i])))/(roots[i]-roots[j]);
      }
    });
  return Chebyshev;
  }
   
   
  } //end of parallel namespace
 } // end of discretization namespace
} // end of qwv namespace
