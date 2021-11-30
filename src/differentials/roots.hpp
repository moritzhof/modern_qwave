#pragma once

#include <ranges>
#include <vector>

namespace qwv{
 namespace discretization{
 
    template<typename T>
    class roots{
    public:
        
        roots() = default;
        roots(std::size_t N) : N_(N){
            result.reserve(N);
         for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
            this->result.push_back(std::cos(M_PI*(2*i+1)/(2*N)));
            }
        }
        
        ~roots() = default;
        
        auto& operator[](auto i ){
            return this->result[i];
        }
        
        void free(){
            return this->result.resize(0);
        }
    private:
        std::size_t N_;
        std::vector<T> result{};
    };
 
 
 template<typename T>
 class roots2D{
 public:
     
     roots2D() = default;
     roots2D(std::size_t N) : N_(N){
         result.reserve(N);
      for(auto i :std::views::iota(std::size_t(0)) | std::views::take(N)){
         this->result.push_back(std::cos(M_PI*(2*i+1)/(4*N)));
         }
     }
     
     ~roots2D() = default;
     
     auto& operator[](auto i ){
         return this->result[i];
     }
     
     void free(){
         return this->result.resize(0);
     }
 private:
     std::size_t N_;
     std::vector<T> result{};
 };
 
#ifdef __CUDACC__
 namespace cuda{
 
    template<typename T>
    class roots{
     
        __device__
        roots(T* roots, std::size_t N): N_(N){
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < N){
                roots[i] = std::cos(M_PI*(2*i+1)/(2*N));
        }
     }
        
    private:
        std::size_t N_;
        std::vector<T> roots{};
        
  };
 }
#endif
    
  } //end of discretization namespace
} // end of qwv namespace
