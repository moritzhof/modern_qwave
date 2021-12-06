#ifdef __CUDACC__

#include "../../cuda/memory.cu"
#include "../operator_base.hpp"
#include "../../operations/cuda/dgemm.cu"

namespace qwv{
 namespace operator{
     namespace cuda{
    


template<typename T>
class kron2D: public OperatorBase<T>
{
public:

  using typename OperatorBase<T>::lidx;
  using typename OperatorBase<T>::gidx;
  using typename OperatorBase<T>::vector;

  friend std::ostream& operator<<<T>(std::ostream&, const kron2D&);

 
  kron2D(int m, qwv::cuda::device_ptr<double> C_n,
         int n, qwv::cuda::device_ptr<double> D_m,
         qwv::cuda::device_ptr<double> V,
         T a1, T a2)
    : C_(C_n), D_(D_m), V_(V), a1_(a1), a2_(a2), n_(n), m_(m) {

    ldC_=n_, ldD_=m_;
    if ( C_.numrows()!=C_.numcols() ||
         D_.numrows()!=D_.numcols() ||
         C_.numrows()!=n_ ||
         D_.numrows()!=m_)
         {
           std::cout << "n_="<<n_<<std::endl;
           std::cout << "m_="<<m_<<std::endl;
           std::cerr << "C is " << C_.numrows() << "x"<<C_.numcols()<<" with ldC="<<ldC_<<std::endl;
           std::cerr << "D is " << D_.numrows() << "x"<<D_.numcols()<<" with ldD="<<ldD_<<std::endl;
           throw std::logic_error("kron2D: C and D should be square matrices of dimension n and m, respectively.");
         }
  }

  //!
void apply(T alpha, const T* v_in, T beta, T* v_out) const
{

 if (alpha*a1_!=static_cast<T>(0)) {
   qwv::cublas::dgemm1(n_, n_, m_, &C_[0], ldC_, v_in, n_, v_out, n_, alpha*a1_, beta);
 }
 else {
#pragma unroll
   for (lidx i=0; i<num_local_rows(); i++) {
     v_out[i]=beta*v_out[i];
   }
 }
 if (alpha*a2_!=static_cast<T>(0)) {
   qwv::cublas::dgemm2(n_, m_, m_, v_in, n_, &D_[0], ldD_, v_out, n_, alpha*a2_, static_cast<T>(1));
 }

if (V_.size()==0) return;

#pragma unroll
    for(lidx i = 0; i < num_local_rows(); i++){
      v_out[i] += alpha*V_[i]*v_in[i];
   }
 }

  // no parallelization yet -> local=global num rows
  inline lidx num_local_rows() const
  {
    return lidx(num_global_rows());
  }

  inline gidx num_global_rows() const
  {
    return gidx(n_*m_);
  }

  inline const dense_matrix& C() const {return C_;}
  inline const dense_matrix& D() const {return D_;}

protected:

  int n_,m_,ldC_,ldD_;
  double a1_, a2_;
  dense_matrix C_;
  dense_matrix D_;
  // diagonal matrix of size n*m representing a potential
  std::vector<T> V_;

};
     } //end of namespace cuda
   } //end of namespace operator
} //end of namespace qwv
#endif
