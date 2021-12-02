#pragma once

#include "../qwv_types.hpp"
#include "../operators/operator_base.hpp"
#include "../operators/operator_tensor.hpp"
#include "../operations/dgemm.hpp"

#include <ostream>

namespace qwv{
namespace preconditioner{
//template<typename ST>
//class Kron2D;
//
//template<typename T>
//class Kron4D;


//! object to solve the Sylvester equation
//!
//! C*X + X*D^T = B, with X, B in R^{n x m}
//!
//! respectively the linear system
//! (I (x) C)x + (D (x) I)x=b, with
//! x=reshape(X,n*m,1), b=reshape(B,n*m,1),
//! and (x) denoting the Kronecker-product.
//!
//! The equation is given as a Kron2D operator type
//! (which is constructed from the components C and D).
//!
//! The V-component of the input Kron2D object is ignored.
//! The Sylvester equation is solved by calling the "apply" function.
//!
template<typename ST>
class SylvesterSolver: public OperatorBase<ST>
{
public:

  using typename OperatorBase<ST>::vector;

  // note: we may want to create a separate operator class
  // for dense matrices (see comment in Kron4D below)
  // Also, we may want to have the option of padding these matrices,
  // i.e. have a 'leading dimension' other than nrows/ncols.
  using dense_matrix = qwv::matrix<ST>;

  SylvesterSolver(qwv::kron2D<ST> const& H)
    : m_(H.C().numrows()), n_(H.D().numrows()),
      CQ_(H.C().numrows(),H.C().numcols()),
      CR_(H.C().numrows(),H.C().numcols()),
      DQ_(H.D().numrows(),H.D().numcols()),
      DR_(H.D().numrows(),H.D().numcols())
  {
    //TODO: compute store Schur-decompositions of a1*C and a2*D'
    return;
  }

// TODO: should solve the Sylvester equation, but right now it will just
//       behave like the identity operator (no preconditioning)
void apply(ST alpha, const ST* v_in, ST beta, ST* v_out) const {
#pragma omp parallel for schedule(static)
  for (int i=0; i<num_local_rows(); i++){
    v_out[i] = alpha*v_in[i] + beta*v_out[i];
  }
 }

  // no parallelization yet -> local=global num rows
  inline lidx num_local_rows() const
  {
    return lidx(m_*n_);
  }

  inline gidx num_global_rows() const
  {
    return gidx(m_*n_);
  }

private:

  lidx m_, n_;
  dense_matrix CQ_, CR_;
  dense_matrix DQ_, DR_;

};

 } //end of preconditioner namespace
} //end of qwv namespace
