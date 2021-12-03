#ifdef __CUDACC__
#include <cuda.h>

namespace qwv {
    namespace cuda{

template<typename T>
class device_ptr{
  T* _p = nullptr;

public:

  __device__ __host__ __inline explicit device_ptr(T* p) : _p(p){}

  template<typename T1, typename = std::enable_if_t<std::is_convertible<T1*, T*>::value>>
  __device__ __host__ device_ptr(const device_ptr<T1>& dp) : _p(dp.get()){}
    
    auto& operator[](std::size_t i){
        return _p[i];
    }
};


template<typename T>
class device_memory{
  T* _p = nullptr;
  device_memory(std::size_t bytes){cudaMallocManaged(&_p, bytes);}

public:
  static device_memory allocate_vector(std::size_t n){return n*sizeof(T);}
  static device_memory allocate_matrix(std::size_t n, std::size_t m){return n*m*sizeof(T);}
  static device_memory allocate_bytes(std::size_t bytes){return {bytes};}
  ~device_memory(){ if(_p){ cudaFree(_p); }}

   operator device_ptr<T>() const{
       return device_ptr<T>(_p);
  }
    
    auto& operator[](std::size_t i){
        return _p[i];
    }

};

template<typename T>
__host__ inline auto make_device_ptr(T* dptr){
  return device_ptr<T>(dptr);
}
    
 int synchronize(){
     return cudaDeviceSynchronize();
 }

 }
}
#endif
