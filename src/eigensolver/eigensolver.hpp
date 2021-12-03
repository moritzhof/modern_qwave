#pragma once

#ifdef HAVE_PHIST
#include "phist_enums.h"
#endif

namespace qwv{
 namespace eigensolver{
  // forward declaration of operator class
  template<typename ST>
  class OperatorBase;


  //! the phist interface is similar to the parpack interface, but we allow the user
  //! to set all solver options in the typical jadaOpts text-file and just give us the
  //! name of the file.
  //
  // A_op: Hamiltonian operator
  // P_op: preconditioner (may be nullptr)
  //
    void phist_interface(OperatorBase<double> *A_op,
        OperatorBase<double> *P_op,
        const char* jadaOpts_filename,
        int k, long int calcEigVec,
        double *dr, double *di,
        double *eigVecReal, double *eigVecIm,
        void const* phist_map=nullptr);


 }
}
