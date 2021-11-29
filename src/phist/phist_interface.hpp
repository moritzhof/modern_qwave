

#include "../operator/operator_base.hpp"

#include <memory>

#ifndef HAVE_PHIST
#error "this file should not be compiled without the -DHAVE_PHIST lag (set by CMake if phist is found)"
#endif

#include "phist_config.h"
#include "phist_kernels.hpp"
#include "phist_core.hpp"
#include "phist_jada.hpp"
#include "phist_MemOwner.hpp"

namespace qwv{

void phist_interface(OperatorBase<double> *A_op,
        OperatorBase<double> *P_op,
        const char* jadaOpts_filename,
        int k,long int calcEigVec,
        double *dr, double *di,
        double *eigVecReal, double *eigVecIm)
{
    //some typedefs that will make it easier to switch to another data type (like complex<double>)
    using ST=double;
    using CT=std::complex<double>;
    using ps=::phist::ScalarTraits<ST>;
    using pk=::phist::kernels<ST>;
    using pc=::phist::core<ST>;
    using pj=::phist::jada<ST>;
    //PHIST interface to calculate eigenvalues of a real, non-symmetric matrix
    //note: we use the same interface here as for parpack to make life easy,
    //      later on it may be useful to expose some solver parameters and improve
    //      the parallel matrix construction and post processing.
    /*
     * k: Number of requested eigenvalues
     * maxit: maximum number of iterations
     * sigma: Eigenvalues close to sigma are calculated -> not implemented yet
     */

    // the iflag is the last argument of almost all phist functions and serves the dual purpose
    // of providing options to the function and letting it report errors back to us.
    // The PHIST_CHK_IERR macro can be used to catch these errors and print error messages or even a
    // simple call stack.
    int iflag=0;

    // options for the Jacobi-Davidson eigensolver
    phist_jadaOpts opts;
    phist_jadaOpts_setDefaults(&opts);
    PHIST_CHK_IERR(phist_jadaOpts_fromFile(&opts, jadaOpts_filename, &iflag),iflag);

    // all iterative solvers in phist take the matrix in form of an abstract operator so that they are
    // not restricted to the internal sparse matrix format.
    phist_DlinearOp *A_phist = QWaves::wrap(A_op);
    phist_DlinearOp *P_phist = nullptr;
    if (P_op)
    {
    P_phist = QWaves::wrap(P_op);
    }

    phist_const_map_ptr map=A_phist->domain_map;
    phist_const_comm_ptr comm=nullptr;
    PHIST_CHK_IERR(phist_map_get_comm(map,&comm,&iflag),iflag);
    MPI_Comm mpi_comm;
    PHIST_CHK_IERR(phist_comm_get_mpi_comm(comm,&mpi_comm,&iflag),iflag);
    int rank, nproc;
    PHIST_CHK_IERR(phist_comm_get_rank(comm,&rank,&iflag),iflag);
    PHIST_CHK_IERR(phist_comm_get_size(comm,&nproc,&iflag),iflag);

    phist_lidx nloc;
    phist_gidx nglob;

    PHIST_CHK_IERR(phist_map_get_local_length(map,&nloc,&iflag),iflag);
    PHIST_CHK_IERR(phist_map_get_global_length(map,&nglob,&iflag),iflag);

    opts.numEigs=k;

    // Q must have at least numEigs+blockSize-1 vectors,
    // but if we reserve more, phist will return a larger
    // basis that we can use to get better eigenvectors or
    // to restart for different parameters
    int nQ=opts.minBas;

    // setup necessary vectors and matrices for the schur form
    ps::mvec_t* Q = NULL;
    PHIST_CHK_IERR(pk::mvec_create(&Q,map,nQ,&iflag),iflag);
  // C++ trick to automatically delete the underlying object at the end of the scope
  // although we'ld have to call a C function to do so.
  phist::MvecOwner<ST> _Q(Q);
  ps::sdMat_t* R = NULL;
  PHIST_CHK_IERR(pk::sdMat_create(&R,nQ,nQ,comm,&iflag),iflag);
  phist::SdMatOwner<ST> _R(R);
  ps::magn_t resNorm[nQ];
  CT z_ev[nQ];


  // if there is a file called phist_Q.bin, read it as starting space
  // otherwise, use a random initial vector v0
  ps::mvec_t* v0 = NULL;
//  if (FILE *file = fopen("phist_Q.bin", "r"))
//        {
//          fclose(file);
//          pk::mvec_read_bin(Q,"phist_Q.bin",&iflag);
//          if (iflag==PHIST_SUCCESS)
 //         {
 //           PHIST_SOUT(PHIST_INFO,"...starting Jacobi-Davidson from phist_Q.bin file. If that is not what you want, remove it from the run directory.");
 //           opts.v0=Q;
 //         }
 //         else
 //         {
 //           PHIST_SOUT(PHIST_INFO,"...there is a phist_Q.bin file in your run directory, but it is ignored because it can't be read (may have the wrong dimensions)");
 //         }
 //   }
 //   else
 //   {
      // setup random start vector
      PHIST_CHK_IERR(pk::mvec_create(&v0,map,1,&iflag),iflag);
      PHIST_CHK_IERR(pk::mvec_random(v0,&iflag),iflag);
      opts.v0=v0;
   // }

    phist::MvecOwner<ST> _v0(v0);

  // used to calculate explicit residuals
  ps::mvec_t* res;
  PHIST_CHK_IERR(pk::mvec_create(&res,map,nQ,&iflag),iflag);

    // will store how many iterations were performed and how many eigenpairs converged
    int nev=opts.numEigs;
    int nit=opts.maxIters;

    double resNorms[nev];
    pj::subspacejada(A_phist, P_phist, opts,
                         Q, R,
                         z_ev, resNorms,
                         &nev, &nit, &iflag);

  if (rank==0)
  {
    if (iflag!=0) std::cout << "non-zero return code from subspacejada: "<<iflag<<std::endl;
    std::cout << "number of converged eigenpairs: "<<nev<<std::endl;
  }
  for (int i=0; i<nev; i++)
  {
    dr[i]=::phist::ScalarTraits<CT>::real(z_ev[i]);
    di[i]=::phist::ScalarTraits<CT>::imag(z_ev[i]);
  }
  for (int i=nev; i<k; i++)
  {
    dr[i]=ST(0);
    di[i]=ST(0);
  }

//  PHIST_CHK_IERR(pk::mvec_write_bin(Q,"phist_Q.bin",&iflag),iflag);

  // for subspacejada, we have to explicitly compute the eigenvectors from the basis Q if we want them:
  if (calcEigVec!=0)
  {
    ps::mvec_t *X=NULL, *AX=NULL;
    PHIST_CHK_IERR(pk::mvec_create(&X, map, nev, &iflag),iflag);
    PHIST_CHK_IERR(pk::mvec_create(&AX, map, nev, &iflag),iflag);
    phist::MvecOwner<ST> _X(X), _AX(AX);
    PHIST_CHK_IERR(pc::ComputeEigenvectors(Q,R,X,&iflag),iflag);
    /* the calling function expects the eigenvectors to be replicated on all MPI ranks right now,
       so we do that manually
     */
    std::unique_ptr<ST> X_localdata(new ST[nloc*nev]);
    PHIST_CHK_IERR(phist_Dmvec_get_data(X, X_localdata.get(), nloc, 0, &iflag), iflag);

    // create count and offset arrays from the local vector lengths.
    // We need this later for the Allgatherv operation (note that not
    // all MPI ranks may have the same nloc).
    int inloc=(int)nloc;
    int counts[nproc], disps[nproc+1];
    PHIST_CHK_IERR(iflag=MPI_Allgather(&nloc, 1, MPI_INT, counts, 1, MPI_INT, mpi_comm),iflag);
    disps[0]=0;
    for (int i=0; i<nproc; i++) disps[i+1]=disps[i]+counts[i];

    for (int j=0; j<nev; j++)
    {
      PHIST_CHK_IERR(iflag=MPI_Allgatherv(X_localdata.get()+j*nloc, nloc, MPI_DOUBLE,
                  eigVecReal+j*nglob, counts, disps, MPI_DOUBLE, mpi_comm),iflag);
    }
  }
  // clean up
  PHIST_CHK_IERR(phist_DlinearOp_destroy(A_phist, &iflag), iflag);
  if (P_phist)
  {
    PHIST_CHK_IERR(phist_DlinearOp_destroy(P_phist, &iflag), iflag);
  }
  return;
}


} // end of namespace qwv
