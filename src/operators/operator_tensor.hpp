
#pragma once

#include "../matrix/matrix_map.hpp"
#include "operator_base.hpp"
#include "../operations/dgemm.hpp"
#include "mkl_trans.h"

#include <ostream>



namespace qwv{

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
  // for dense matrices (see comment in kron4D below)
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
           throw std::logic_error("kron2D: C and D should be square matrices of dimension n and m, respectively.");
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

#ifdef HAVE_PHIST
template<typename ST>
class kron4D : public OperatorBase<ST>
{
public:

  //! construct from two kron2D objects C and D,
  //! a diagonal potential matrix V, and scalars a1, a2:
  //! kron4D = a1*(I (x) C) + a2*(D (x) I) + V
  kron4D(const Kron2D<ST>& C, const Kron2D<ST>& D,
        std::vector<double> const& V, ST a1, ST a2, MPI_Comm comm=MPI_COMM_WORLD) :
  C_(C), D_(D), V_(V), a1_(a1), a2_(a2),
  map_(std::tuple{C_.C().numrows(),C_.D().numrows()},std::tuple{D_.C().numrows(),D_.D().numrows()},comm),
 tmap_(std::tuple(D_.C().numrows(),D_.D().numrows()),std::tuple(C_.C().numrows(),C_.D().numrows()),comm)
 {
    if (C_.num_local_rows()!=C_.num_global_rows() ||
        D_.num_local_rows()!=D_.num_global_rows() ||
        map_.map12().num_local_elements()!=map_.map12().num_global_elements() )
        {
          throw std::logic_error("cannot handle distributed component Kron2D objects in kron4D, yet");
        }

        if (map_.map12().num_local_elements()!=C_.num_local_rows()||
            map_.map34().num_global_elements()!=D_.num_global_rows() )
        {
          throw std::logic_error("unexpected dimensions in components C and D of kron4D");
        }
 }

 // move constructor is allowed
 kron4D(kron4D&&)=default;
 // copy constructor is allowed
 kron4D(kron4D const&)=default;
 // destructor is trivial
 ~kron4D()=default;

  //! apply operator to a vector (vectorized 4-tensor).

  //! given a vectorized 4-tensor, apply operator: v_out = alpha*(*this)*v_in+beta*v_out.
  //! If v_in = vec(X), resp. X=reshape(v_in,n,m), the operation is
  //! implemented as Y <- alpha(a1*C*X + a2*X*transpose(D))+beta*Y, v_out=vec(Y).
  //!
  //! Distributed memory:
  //!
  //! The first two dimensions are not split up in our implementation, but dimensions 3 and 4 may be.
  //! In terms of the above notation this means that X, Y are n x m, with n=n1*n2, m=n3*n4, and their
  //! columns may be distributed over the processes. The first term, C*X, is therefore trivially parallel,
  //! but the second requires communication.
  void apply(ST alpha, const ST* v_in, ST beta, ST* v_out)  const
  {
    // some constants
    const ST one=static_cast<ST>(1);
    const ST zero=static_cast<ST>(0);

    // The input/output tensors are interpreted as matrices with n rows
    // and m columns in total (mloc columns on this process). The leading
    // dimension (column stride) is ldX=n.
    const lidx n    = map_.map12().num_global_elements();
    const lidx m    = map_.map34().num_global_elements();
    const lidx mloc = map_.map34().num_local_elements();

    // The transposed (matricized) tensor has m rows and n columns in total
    // (nloc columns on this process). The leading dimension is ldXt=m.
    const lidx nloc = tmap_.map34().num_local_elements();

    // create some temporary matrices
    std::unique_ptr<ST[]> Y1{new ST[n*mloc]};
    std::unique_ptr<ST[]> Xt{new ST[m*nloc]};
    std::unique_ptr<ST[]> Y2t{new ST[m*nloc]};

    std::span<const ST> X_span(v_in,n*mloc);
    std::span<ST> Y_span(v_out,n*mloc);

    std::span<ST> Xt_span(Xt.get(),m*nloc);
    std::span<ST> Y2t_span(Y2t.get(),m*nloc);

    // (1) Start the global transpose of X for the second term.
    auto [requests,xbuf]=start_transpose(X_span,map_,alpha*a2_,Xt_span,tmap_);

    // (2) overlap the communication with the second term: Y2 = alpha*a2*X*D'.
    // The second term is tricky because (X*D') with D a Kron2D can't be expressed directly in matrix-matrix products.
    // We implement it as X*D' = (D*X')'. and compute it column-by column as messages
    // from the transpose come in. The back-transpose will be overlapped with the computation of the
    // first term, which doesn't involve any communication.

    // copy beta*v_out to Y1 if needed
    if (beta!=zero)
    {
#pragma omp parallel for schedule(static)
       for(lidx i = 0; i < num_local_rows(); ++i)
       {
         Y1[i] = v_out[i];
       }
    }

    // wait for the global transpose of alpha*a2*X into Xt
    int index=0;
    while (index!=MPI_UNDEFINED)
    {
      MPI_Waitany(requests.size(), &(requests[0]), &index, MPI_STATUS_IGNORE);
      int i=index-tmap_.map34().offset(); //local column index
      if (i<0 || i>=nloc) continue;
      D_.apply(one,Xt.get()+i*m,zero,Y2t.get()+i*m);
    }
    // start transpose to get Y = Y2t'
    xbuf=nullptr;
    auto [requests2,xbuf2]=start_transpose(Y2t_span,tmap_,one,Y_span,map_);

    // ... overlap with the computation of the first term Y1=alpha*C*X+beta*Y_in.
    // This is easy, but note that the apply function takes
    // only one column of X and Y at a time.
    for (lidx i=0; i<mloc; i++){
      C_.apply(alpha*a1_,v_in+i*n,beta,Y1.get()+i*n);
      }

      // complete transpose to get v_out = Y2
      MPI_Waitall(requests2.size(),&(requests2[0]),MPI_STATUSES_IGNORE);
      xbuf2=nullptr;

      // add Y1+alpha*V*X
      if (V_.size()>0)
      {
#pragma omp parallel for schedule(static)
        for(lidx i = 0; i < num_local_rows(); ++i)
        {
          v_out[i] += alpha*V_[i]*v_in[i] + Y1[i];
        }
      }
      else
      {
#pragma omp parallel for schedule(static)
        for(lidx i = 0; i < num_local_rows(); ++i)
        {
          v_out[i] += Y1[i];
        }
      }
    }

  inline lidx num_local_rows() const
  {
    return map_.num_local_elements();
  }

  inline gidx num_global_rows() const
  {
    return map_.num_global_elements();
  }

  TensorMap4D const& map() const {return map_;}
  TensorMap4D const& tmap() const {return tmap_;}

protected:

  double a1_, a2_;
  Kron2D<ST> C_;
  Kron2D<ST> D_;
  std::vector<ST> V_;

  //! range map.

  //! This object represents the index range (i1,i2,i3,i4) over
  //! all local elements of the input and result vectors of apply
  //! (it is both the 'range' and 'domain' map in Trilinos nomenclature).
  //! In case of distributed memory, the i3 and i4 ranges may be distributed
  //! over multiple processors.
  TensorMap4D map_;

  //! the 'transposed map' which allows iterating over local elements of
  //! the transposed input tensor in the sense that if X=reshape(x_in, n1*n2,n3*n4),
  //! and Xt=transpose(X), tmap iterates over the local elements of Xt.
  TensorMap4D tmap_;

// we make this function public so we can call it in the tests
public:

  //! Start computing Xt = alpha*transpose(X)

  //! where X lives in map, and Xt in tmap. The function returns an array
  //! (std::vector) of MPI_Request objects, one for each column of Xt. Before
  //! accessing column j of Xt, you have to finish the communication for that
  //! column by calling MPI_Wait(rquest[i],...). To finish the complete transpose,
  //! use MPI_Waitall. The second argument is a std::unique_ptr to a temporary buffer
  //! that should be kept until after all requests are dealt with using either MPI_Waitall
  //! or individual MPI_Wait or MPI_Waitany calls.
  auto start_transpose(std::span<ST const> X,  TensorMap4D const& map,  ST alpha,
                       std::span<ST>       Xt, TensorMap4D const& tmap) const
  {
    // temporary storage for the local transpose of a tensor
    std::unique_ptr<ST[]> X_buf = std::unique_ptr<ST[]>(new ST[map.num_local_elements()]);

    // X is n x mloc on this process.
    // X_buf is mloc x n.
    lidx n    = map.map12().num_local_elements();
    lidx mloc = map.map34().num_local_elements();

    // Xt is m x nloc on this process.
    lidx m    = tmap.map12().num_local_elements();
    lidx nloc = tmap.map34().num_local_elements();

    if (std::size(X)!=n*mloc || std::size(Xt)!=m*nloc )
    {
      throw std::logic_error("input and/or output of start_transpose have mismatched extent");
    }

    MPI_Comm comm=map.comm();
    MPI_Datatype dtype=MPI_DOUBLE;

    // first transpose local part, X_buf=alpha*X'.
    // Since this is a memory-bound operation, we do the scaling here 'for free'
    if constexpr(std::is_same<ST,double>::value)
    {
      dtype=MPI_DOUBLE;
#ifdef PHIST_HAVE_MKL
      MKL_Domatcopy('C', 'T', n, mloc, alpha, X.data(), n, X_buf.get(), mloc);
#else
      // available in OpenBLAS
      cblas_domatcopy(CblasColMajor, CblasTrans, n, mloc, alpha, X.data(), n, X_buf.get(), mloc);
#endif
    }
    else if constexpr(std::is_same<ST,std::complex<double>>::value)
    {
      dtype=MPI_DOUBLE_COMPLEX;
      MKL_Zomatcopy('C', 'T', n, mloc, alpha, X.data(), n, X_buf.get(), mloc);
    }
    else
    {
      throw std::logic_error("start_transpose only implemented for double and complex<double> types so far.");
    }

    // then start the MPI communication to get my columns of Xt:

    // note: since we're using collective communication, we need to
    // create a request and start an MPI_Alltoallv for every column
    // of the output Xt.
    std::vector<MPI_Request> requests(n,MPI_REQUEST_NULL);

    int target_proc=0;
    const TensorMap2D& tcols = tmap.map34();

    const int* rcounts = map.map34().counts();
    const int* rdisps = map.map34().disps();

    for (int i=0; i<n; i++)
    {
        if (i >= tcols.disp(target_proc+1)) target_proc++;
        int iloc = tcols.rank()==target_proc? i-tcols.offset(): 0;
        MPI_Igatherv(X_buf.get()+i*mloc, mloc, dtype,
                   Xt.data()+iloc*m, rcounts, rdisps, dtype,
                   target_proc, comm, &(requests[i]));
    }
    return std::make_tuple(requests, std::move(X_buf));
  }

};

template<typename ST>
std::ostream& operator<<(std::ostream& os, const Kron2D<ST>& op)
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

#endif

} // end of namespace qwv
