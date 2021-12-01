
#pragma once

#include "operator_base.hpp"
#include "../operations/dgemm.hpp"

#include <ostream>



namespace qwv{

#include "mkl_trans.h"

template<typename ST>
class kron2D;

template<typename T>
class kron4D;


template<typename ST>
std::ostream& operator<<(std::ostream& os, const kron2D<ST>& op);

// class representing a '2D' operator of the form
//
// a1*(I_m (x) C_n) + a2*(D_m (x) I_n) + V
//
// Where (x) denotes the Kronecker product,
// I_k is a k \times k identity matrix,
// C and D are square matrices with the subscript
// indicating the dimension.
// a1 and a2 are scalars, and
// V is a diagonal matrix representing the potential.
//
// The template parameter indicates the type of the scalar
// entries of C and D, and of the vector this operator can
// be applied to.
template<typename ST>
class kron2D: public OperatorBase<ST>
{
public:

  using typename OperatorBase<ST>::lidx;
  using typename OperatorBase<ST>::gidx;
  using typename OperatorBase<ST>::vector;

  friend std::ostream& operator<<<ST>(std::ostream&, const kron2D&);


  // note: we may want to create a separate operator class
  // for dense matrices (see comment in Kron4D below)
  // Also, we may want to have the option of padding these matrices,
  // i.e. have a 'leading dimension' other than nrows/ncols.
  using dense_matrix = matrix<ST>;

  kron2D(int m, const dense_matrix& C_n,
         int n, const dense_matrix& D_m,
         std::vector<ST> const& V,
         ST a1, ST a2)
    : C_(C_n), D_(D_m), V_(V), a1_(a1), a2_(a2),
      n_(n), m_(m)
  {
    // note: the 'leading dimension' can be larger than the number of rows/columns
    //       of the densematrices C and D (padding). This allows aligned access to
    //       the matrix elements even if n and/or m are not a multiple of the SIMD width.
    //       For now we just use powers of 2 so we can use n, m as the leading dimension,
    //      otherwise we'ld have to adjust the matrix building routines.
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
           throw std::logic_error("Kron2D: C and D should be square matrices of dimension n and m, respectively.");
         }
  }

   //! apply this linear operator to a vector.

   //! given a vectorized 2-tensor, apply operator: v_out = alpha*(*this)*v_in + beta*v_out.
   //!
  //! If v_in = vec(X), resp. X=reshape(v_in,n,m), the operation is
  //! implemented as Y <- alpha*[a1*C*X + a2*X*transpose(D)]+beta*Y, v_out=vec(Y).
  //!
  //! explanation in matlab syntax:
  //!
  //! let X=reshape(x,n,m)
  //! (just reinterpreting, no data movement involved)
  //!
  //! y = Op*x = reshape(a1*C*X + a2*X*D', n*m, 1)
  //!
void apply(ST alpha, const ST* v_in, ST beta, ST* v_out) const
{

 if (alpha*a1_!=static_cast<ST>(0)) {
   dgemm1(n_, n_, m_, &C_[0], ldC_, v_in, n_, v_out, n_, alpha*a1_, beta);
 }
 else {
#pragma omp parallel for schedule(static)
   for (lidx i=0; i<num_local_rows(); i++) {
     v_out[i]=beta*v_out[i];
   }
 }
 if (alpha*a2_!=static_cast<ST>(0)) {
   dgemm2(n_, m_, m_, v_in, n_, &D_[0], ldD_, v_out, n_, alpha*a2_, static_cast<ST>(1));
 }

if (V_.size()==0) return;

#pragma omp parallel for schedule(static)
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
  std::vector<ST> V_;

};

// class representing a '4D' operator of the form
//
// (D (x) I_1) (x) I_2 + I_2 (x) (D (x) I_1) + ...
// (I_1 (x) D) (x) I_2 + I_2 (x) (I_1 (x) D)
//
// Where (x) denotes the Kronecker product,
// I_1 (I_2) are n (n*n)-dimensional unit matrices,
// D is square and m x m,
// indicating the dimension.
//
// The template parameter indicates the type of the scalar
// entries of C and D, and of the vector this operator can
// be applied to.
template<typename ST>
class kron4D : public OperatorBase<ST>
{
public:

  using typename OperatorBase<ST>::lidx;
  using typename OperatorBase<ST>::gidx;

  //! construct from two Kron2D objects C and D,
  //! a diagonal potential matrix V, and scalars a1, a2:
  //! Kron4D = a1*(I (x) C) + a2*(D (x) I) + V
  kron4D(const kron2D<ST>& C, const kron2D<ST>& D,
        std::vector<double> const& V, ST a1, ST a2) :
  C_(C), D_(D), V_(V), a1_(a1), a2_(a2)
 {
 }

  //! apply operator to a vector (vectorized 4-tensor).

  //! given a vectorized 4-tensor, apply operator: v_out = (*this)*v_in.
  //! If v_in = vec(X), resp. X=reshape(v_in,n,m), the operation is
  //! implemented as Y <- a1*C*X + a2*X*transpose(D), v_out=vec(Y).
  void apply(ST alpha, const ST* v_in, ST beta, ST* v_out)  const
  {

    // some constants
    const ST one=static_cast<ST>(1);
    const ST zero=static_cast<ST>(0);
    const lidx n = C_.num_local_rows();
    const lidx m = D_.num_local_rows();
    const lidx ldX=n, ldXt=m;

    //... and some temporary matrices
    ST* Xt = new ST[n*m];
    ST* Y1 = new ST[n*m];
    ST* Y2 = new ST[n*m];

    if (beta!=zero)
    {
#pragma omp parallel for schedule(static)
       for(lidx i = 0; i < num_local_rows(); ++i)
       {
         Y1[i] = v_out[i];
       }
    }

    // first term: Y1=alpha*a1*C*X, where C is a Kron2D.
    // This is easy, but note that the apply function takes
    // only one column of X and Y at a time.
    for (lidx i=0; i<m; i++){
      C_.apply(alpha*a1_,v_in+i*n,beta,Y1+i*n);
      }

    // Second term: Y2 = alpha*a2*X*D'. This is tricky because
    // (X*D') with D a Kron2D can't be expressed directly in matrix-matrix products.
    //
    // We implement it as X*D' = (D*X')', at least for now.
    // For a faster implementation see Paolo Bientinesi's 2016 GETT-paper ("GEMM-like
    // Tensor-Tensor multiplication").
    //
    // Note that we can't just call different GEMM-routines because
    // the matrix X is column-major, but our Kron2D::apply-algorithm
    // would have to again reshape each column of X' (row of X) into
    // a matrix to do the GEMMs.

    // compute Y2=a2*alpha*D*X'

    // first transpose, X2=alpha*a2*X'.
    // since this is a memory-bound operation, we do the scaling here 'for free'
    MKL_Domatcopy('C', 'T', n, m, alpha*a2_, v_in, ldX, Xt, ldXt);

    // then apply, Y2 = D*X2 = a2*alpha*D*X'
    for (lidx i=0; i<n; i++){
      D_.apply(one,Xt+i*m,zero,Y2+i*m);
      }

    // finally transpose-add to final result: Y = Y1+Y2'

    //MKL_Domatadd('C', 'N', 'T', m, n, one, Y1, ldX, one, Y2, ldXt, v_out, ldX);
    //TODO: replace this ad-hoc implementation by the above MKL call once we figure out
    //      why it doesn't behave correctly
#pragma omp parallel for schedule(static)
    for (lidx j=0; j<m; j++){
      for (lidx i=0; i<n; i++){
        v_out[j*ldX+i]=Y1[j*ldX+i]+Y2[i*ldXt+j];
        }
      }

    delete [] Xt;
    delete [] Y1;
    delete [] Y2;

    if (V_.size()>0)
    {
#pragma omp parallel for schedule(static)
       for(lidx i = 0; i < num_local_rows(); ++i){
       v_out[i] += alpha*V_[i]*v_in[i];
       }
    }
  }

  // no parallelization yet -> local=global num rows
  inline lidx num_local_rows() const
  {
    return lidx(num_global_rows());
  }

  inline gidx num_global_rows() const
  {
    return C_.num_global_rows()*D_.num_global_rows();
  }

protected:

  double a1_, a2_;
  kron2D<ST> C_;
  kron2D<ST> D_;
  std::vector<ST> V_;
};

template<typename ST>
std::ostream& operator<<(std::ostream& os, const kron2D<ST>& op)
{
  os << "% BEGIN MATLAB/OCTAVE CODE"<<std::endl;
  os << "% Operator a1*(I_m (x) C_n) + a2*(D_m (x) I_n) + V"<<std::endl;
  os << "a1="<<op.a1_<<std::endl;
  os << "a2="<<op.a2_<<std::endl;
  os << "m="<<op.m_<<std::endl;
  os << "n="<<op.n_<<std::endl;
  os << "C_n=["<<op.C_<<"];"<<std::endl;
  os << "D_m=["<<op.D_<<"];"<<std::endl;
  os << "V=[";
  for (auto v: op.V_) os << v << " ";
  os << "];"<<std::endl;
  os << "% END MATLAB/OCTAVE CODE"<<std::endl;
  return os;
}

} // end of namespace qwv
