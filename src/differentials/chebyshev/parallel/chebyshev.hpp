
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
   
   template<typename T>
   matrix<T> buildDifferentialMatrixChebD2(std::size_t N, matrix<T>& ChebD1){
     //auto v = std::views::iota(static_cast<std::size_t>(0), N);
     std::vector<T> x_Cheb(N);
     matrix<T> ChebD2(N,N);

     //std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
     //std::for_each(counting_iterator(0), counting_iterator(N), [&](int i){
      for(int i = 0; i < N; ++i){
       x_Cheb[i] = cos(PI*(2*i+1)/(2*N));
     }

     //std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
     //std::for_each(counting_iterator(0), counting_iterator(N), [&](int i){
      for(int i = 0; i < N; ++i){
       for(auto j = 0; j < N; ++j){
         if(i == j){
               ChebD2[i*N+j] = pow(x_Cheb[i],2)/pow(1.0-pow(x_Cheb[i],2),2)-(pow(N,2)-1)/(3.0*(1.0-pow(x_Cheb[i],2)));
         }
         else {
              ChebD2[i*N+j] = ChebD1[i*N+j]*(x_Cheb[i]/(1.0-pow(x_Cheb[i],2))-2.0/(x_Cheb[i]-x_Cheb[j]));
          }
       }
     }

     return ChebD2;
   }

   

   //this will only use vectors: That is, only store the Diagonal Elements
   template<typename T>
   matrix<T> buildDifferentialMatrixTBChebD1(std::size_t N, double L, matrix<T>& ChebD1){

     //auto v = std::views::iota(static_cast<std::size_t>(0), N);
     std::vector<T> x_Cheb(N);
     matrix<T> A11(N,N), TBChebD1(N,N);

     //std::for_each(std::execution::par, std::begin(v), std::end(v),[&](auto i){
     //std::for_each(counting_iterator(0), counting_iterator(N), [&](int i){
     for(int i = 0; i < N; ++i){
       x_Cheb[i] = std::cos(PI*(2*i+1)/(2*N));
      }
     for(int i = 0; i < N; ++i){
       A11[i*N+i] = 1.0/L*std::pow(std::sqrt(1.0-std::pow(x_Cheb[i],2)),3);
     }

    //DONE -- TO DO: A11 is just a diagonal mat. no need to store the zeros. switch to 1D vector and modify operator*
    //       or just implement the operation with a basic loop
    //std::for_each(std::execution::par, std::begin(v), std::end(v),[&](auto i){
    //std::for_each( counting_iterator(0), counting_iterator(N), [&](int i){
   //for(int i = 0; i < N; ++i){
    //  for(int j = 0; j < N; ++j){
        TBChebD1 = A11*ChebD1;
     // }
    // }

     return TBChebD1;

   }

   template<typename T>
   matrix<T> buildDifferentialMatrixTBChebD2(std::size_t N, double L, matrix<T>& ChebD1, matrix<T>& ChebD2){
     std::vector<T> x_Cheb(N);
     matrix<T> TBChebD2(N,N);
     matrix<T> A22(N,N), A21(N,N);
     matrix<T> temp(N,N);

     for(int i = 0; i < N; ++i){
         x_Cheb[i] = std::cos(PI*(2*i+1)/(2*N));
         }

     for(int i = 0; i < N; ++i){
         A21[i*N+i]=-3.0/std::pow(L,2)*x_Cheb[i]*std::pow(1.0-std::pow(x_Cheb[i],2),2);
         A22[i*N+i]=1.0/std::pow(L,2)*std::pow(1.0-std::pow(x_Cheb[i],2),3);
       }

       TBChebD2 = A22*ChebD2;
       temp = A21*ChebD1;

       TBChebD2 = temp+TBChebD2;

       return TBChebD2;

   
  } //end of parallel namespace
 } // end of discretization namespace
} // end of qwv namespace
