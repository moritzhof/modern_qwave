
#include "qwv_util.hpp"


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

 namespace parallel{
 template<typename T>
 auto constexpr Chebyshev1D(std::size_t& N){

    auto v = std::views::iota(std::size_t(0), N);
    auto result = qwv::discretization::roots<double>(N);
    auto Chebyshev = std::vector<double>(N*N,0);

   std::for_each(std::execution::par_unseq, std::begin(v), std::end(v), [&](auto i){
     for(auto j :std::views::iota(std::size_t(0)) | std::views::take(N)){
       (i == j) ?
         Chebyshev[i*N+i] = 0.5*result[i]/(1.0-(result[i]*result[i]))
       :
         Chebyshev[i*N+j] = std::cos(M_PI*(i+j))*std::sqrt((1.0-(result[j]*result[j]))/(1.0-(result[i]*result[i])))/(result[i]-result[j]);
      }
    });
     
   //result.free();
   return Chebyshev;
 }
 
 
 
  }
 }
}






auto main()->int{

std::size_t N = 10;
//auto cheby = util::to_vector(qwv::discretization::Chebyshev<double>(N));
auto cheby1D = qwv::discretization::Chebyshev1D<double>(N);
auto cheby2D = qwv::discretization::Chebyshev2D<double>(N);
    
auto par_cheby1D = qwv::discretization::parallel::Chebyshev1D<double>(N);

    for(const auto& i : cheby1D){
        std::cout<< i << ' ';
    }
    std::cout<<'\n';
    std::cout << " ###### COMPLETED: PARALLEL 1D ######## \n";
    for(const auto& i : par_cheby1D){
        std::cout<< i << ' ';
    }
    std::cout<<'\n';
    
    std::cout << " ###### COMPLETED: 1D ######## \n";
    for(const auto& i : cheby2D){
      std::cout<< i << ' ';
    }
    std::cout<<'\n';

    std::cout << " ###### COMPLETE ######## \n";
return 0;
}
