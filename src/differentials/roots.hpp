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
    
  } //end of discretization namespace
} // end of qwv namespace
