
#pragma once
#include "../operations/dgemm.hpp"
#include "../matrix/matrix.hpp"

#ifdef HAVE_PHIST
#include "phist_config.h"
#include "phist_ScalarTraits.hpp"
#include "phist_kernels.hpp"
#include "phist_core.hpp"
#endif

namespace qwv
{

  template<typename ST>
  class OperatorBase
  {
  public:

#ifdef HAVE_PHIST
    // local index type
    using lidx=phist_lidx;
    // global index type
    using gidx=phist_gidx;
#else
    // local index type
    using lidx=int;
    // global index type
    using gidx=long long int;

#endif

     using vector=std::vector<ST>;
     
    //! compute vecOut = alpha*vecIn + beta* vecOut
   
    virtual  void apply(ST alpha, const double* vec_in, ST beta, double* vec_out) const = 0;
        
    virtual lidx num_local_rows() const = 0;

    virtual gidx num_global_rows() const = 0;
  };

#ifdef HAVE_PHIST

  namespace detail
  {

    template<typename ST>
    void apply_shifted_OperatorBase(ST alpha, const void* v_op, ST const * sigma,
        typename ::phist::ScalarTraits<ST>::mvec_t const* vec_in, ST beta,  typename ::phist::ScalarTraits<ST>::mvec_t* vec_out, int* iflag)
    {
      using st=::phist::ScalarTraits<ST>;
      *iflag=0;
      const OperatorBase<ST>* op = (const OperatorBase<ST>*)v_op;

      int ncols_in, ncols_out;
      PHIST_CHK_IERR(::phist::kernels<ST>::mvec_num_vectors(vec_in,&ncols_in,iflag),*iflag);
      PHIST_CHK_IERR(::phist::kernels<ST>::mvec_num_vectors(vec_out,&ncols_out,iflag),*iflag);
      phist_lidx ldX_in, ldX_out;
     
     ST const* v_in =  nullptr;
     ST* v_out = nullptr;

      // get a raw pointer and the column-stride from he multi-vector objects.
      PHIST_CHK_IERR(::phist::kernels<ST>::mvec_extract_const_view(vec_in, &v_in, &ldX_in, iflag),*iflag);
      PHIST_CHK_IERR(::phist::kernels<ST>::mvec_extract_view(vec_out, &v_out, &ldX_out, iflag), *iflag);

#ifdef PHIST_MVECS_ROW_MAJOR
      // we can't handle multi-vectors in row-major ordering except in very special cases right now
      PHIST_CHK_IERR(*iflag=(ncols_in!=1 || ldX_in!=1 || ldX_out!=1)? PHIST_NOT_IMPLEMENTED: 0, *iflag);
#endif
      PHIST_CHK_IERR(*iflag=(ncols_in!=ncols_out)? PHIST_INVALID_INPUT: 0, *iflag);

      // to avoid forcing every class derived from OperatorBase to implement this
      // very Jacobi-Davidson specific function, and since the operators in this project
      // seem to be compute-bound, we implement the shifting in this general but sub-optimal
      // way here
      double beta_prime=beta;
      if (sigma!=nullptr)
      {
        ST alpha_prime[ncols_in];
        for (int j=0; j<ncols_in; j++) alpha_prime[j]=-alpha*sigma[j];
        PHIST_CHK_IERR(::phist::kernels<ST>::mvec_vadd_mvec(alpha_prime,vec_in, beta, vec_out, iflag), *iflag);
        beta_prime=ST(1);
      }

      try {
        for (int j=0; j<ncols_in; j++)
        {
          op->apply(alpha, v_in+j*ldX_in, beta_prime, v_out+j*ldX_out);
        }
      } catch (std::exception& e)
      {
        std::cerr << e.what() << std::endl;
        *iflag=PHIST_CAUGHT_EXCEPTION;
      }
      return;
    }

    template<typename ST>
    void apply_OperatorBase(ST alpha, void const* v_op, typename ::phist::ScalarTraits<ST>::mvec_t const* vec_in, ST beta, typename ::phist::ScalarTraits<ST>::mvec_t* vec_out, int *iflag)
    {
      PHIST_CHK_IERR(apply_shifted_OperatorBase(alpha, v_op, (ST*)nullptr, vec_in, beta, vec_out, iflag), *iflag);
    }
  }//namespace detail

  // given an OperatorBase, creates the corresponding phist object that can
  // be passed to the phist iterative solvers
  template<typename ST>
  typename phist::ScalarTraits<ST>::linearOp_t* wrap(OperatorBase<ST>* q_op)
  {
    int iflag=0;
    typename ::phist::ScalarTraits<ST>::linearOp_t* op=new typename ::phist::ScalarTraits<ST>::linearOp_t;
    op->A=(void*)q_op;
    op->apply=&(detail::apply_OperatorBase);
    op->apply_shifted=&(detail::apply_shifted_OperatorBase);
    phist_comm_ptr comm=nullptr;
    phist_comm_create(&comm,&iflag);
    if (iflag!=0) throw std::runtime_error("phist_comm_create returned non-zero error code");
    phist_map_ptr map=nullptr;
    phist_lidx lnrows = q_op->num_local_rows();
    phist_gidx gnrows = q_op->num_global_rows();
    phist_map_create(&map,comm,gnrows,&iflag);
    if (iflag!=0) throw std::runtime_error("phist_map_create returned non-zero error code");
    // check that the given operator is compatible with this map.
    // unfortunately there is currently no general interface function to
    // create a map with a given distribution in phist, so if it doesn't match
    // we have to either
    //  - use the phist default map before to get the same distribution,
    //  - add a phist kernel function that takes both nloc and nglob
    //  - directly construct the underlying data structure, encapsulated in
    //    e.g. #ifdef PHIST_KERNEL_LIB_EPETRA|GHOST|...
    phist_lidx map_nloc;
    phist_gidx map_nglob;
    phist_map_get_global_length(map,&map_nglob,&iflag);
    phist_map_get_local_length(map,&map_nloc,&iflag);
    if (lnrows!=map_nloc ||gnrows!=map_nglob)
    {
      throw std::logic_error("the phist default distribution (map) doesn't match the given operator's row distribution");
    }
    op->range_map=map;
    op->domain_map=map;

    // set all the members we don't need to NULL to prevent phist from trying to call/access them
    op->aux=nullptr;
    op->use_transpose=false;
    op->shifts=nullptr;
    op->applyT=nullptr;
    op->fused_apply_mvTmv=nullptr;
    op->update=nullptr;
    op->destroy=nullptr;
    return op;
  }
#endif /* HAVE_PHIST */
}// namespace Qqwv
