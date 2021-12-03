
#pragma once

#include "../qwv_types.hpp"
#include "../coord/coord.hpp"

#include <memory>
#include <mpi.h>


class TensorMap2D{
private:

  using iter=qwv::coord;
  using pos_type=iter::pos_type;
    
public:

  TensorMap2D(int n1, int n2, MPI_Comm comm=MPI_COMM_SELF);
  TensorMap2D() = default;
  TensorMap2D(TensorMap2D const&)=default;
  ~TensorMap2D()=default;

  iter begin() const {return iter(offset_,std::tuple{n1_,n2_});}

  iter end() const {return begin()+num_local_elements();}

  // note: this will throw an exception if element (i,j) is not on the present MPI process in comm
  lidx operator()(int i, int j) const {return lid(i,j);}
  
  lidx lid(int i, int j) const {
    pos_type pos=iter(std::tuple{i,j},std::tuple{n1_,n2_}).pos;
    lidx idx=lidx(pos-offset_);
    if (idx<0||idx>=local_len_) throw std::runtime_error("iterator out-of-bounds");
    return idx;
  }

  gidx gid(int i, int j) const {
    pos_type pos=iter(std::tuple{i,j},std::tuple{n1_,n2_}).pos;
    return gidx(pos);
  }

  lidx num_local_elements() const { return local_len_;}
  lidx num_global_elements() const { return n1_*n2_;}
    
    
private:
  int n1_, n2_;
  lidx offset_, local_len_;
  MPI_Comm comm_;
  int rank_, nproc_;
  std::shared_ptr<int[]> counts_;
  std::shared_ptr<int[]> disps_;
};


TensorMap2D::TensorMap2D(int n1, int n2, MPI_Comm comm)
 : n1_(n1), n2_(n2), comm_(comm), counts_(nullptr), disps_(nullptr)
{
  MPI_Comm_rank(comm_,&rank_);
  MPI_Comm_size(comm_,&nproc_);
  counts_=std::shared_ptr<int[]>(new int[nproc_]);
  disps_=std::shared_ptr<int[]>(new int[nproc_+1]);
  lidx chunk= (lidx(n1_)*lidx(n2_))/nproc_;
  lidx rem  = (lidx(n1_)*lidx(n2_))%nproc_;
  disps_[0]=0;
  for (int p=0; p<nproc_; p++)
  {
    counts_[p] = chunk + (p<rem? 1: 0);
    disps_[p+1]= disps_[p]+counts_[p];
  }
  offset_=disps_[rank_];
  local_len_=counts_[rank_];
}


#ifdef HAVE_PHIST
#include "phist_config.h"
#include "phist_kernels.h"
#include "phist_MemOwner.hpp"

class TensorMap4D
{

private:

  using iter=qwv::MultiIndex4D;
  using quad=qwv::MultiIndex4D::quad;
  using pos_type=qwv::MultiIndex4D::pos_type;

public:

  //! Create a data distribution strategy for a tensor of dimension
  //! n=n1 x n2 x n3 x n4, where the first dimension n1 is the 'fastest index'
  //! in the sense that elements X(i,j,k,l) and  X(i+1,j,k.l) are in adjacent
  //! memory locations.
  //!
  //! We do not split up the first two dimensions among processes, so that a complete
  //! (:,:,i,j) slice is always local.
  //!
  //! Example:
  //!
  //! gidx n1,n2,n3,n4;
  //! TensorMap4D map(std::tie(i1,i2),std::tie(i3,i4), MPI_COMM_WORLD);
  //!
  TensorMap4D(std::tuple<int,int> n12, std::tuple<int,int> n34, MPI_Comm comm=MPI_COMM_WORLD);

  TensorMap4D() = delete;
  TensorMap4D(TensorMap4D const&)=default;
  ~TensorMap4D()=default;

  iter begin() const {return iter(offset_,std::tie(n1_,n2_,n3_,n4_));}

  iter end() const   {return begin()+num_local_elements();}

  //! operator(i,j,k,l). same as lid(i,j,k,l).
  lidx operator()(int i, int j, int k, int l) const {return lid(i,j,k,l);}

  //! returns a linear local index into a 4-tensor
  //! note: this will throw an exception if element (i,j) is not on the present MPI process in comm
  lidx lid(int i, int j, int k, int l) const
  {
    pos_type idx=iter(std::tie(i,j,k,l),std::tie(n1_,n2_,n3_,n4_)).pos-offset_;
    if (idx<0||idx>num_local_elements()) throw std::runtime_error("iterator out-of-bounds");
    return lidx(idx);
  }

  //! returns a linear global index into a 4-tensor
  gidx gid(int i, int j, int k, int l) const {return gidx(iter(std::tie(i,j,k,l),std::tie(n1_,n2_,n3_,n4_)).pos);}

  //! returns the number of global elements in the tensor
  gidx num_global_elements() const {return gidx(n1_)*gidx(n2_)*gidx(n3_)*gidx(n4_);}

  //! returns the number of local elements
  lidx num_local_elements() const {return local_len_;}

  //! get global dimensions of the tensor

  //! usage auto [n1,n2,n3,n4]=global_dims()
  quad global_dims() const {return std::tie(n1_,n2_,n3_,n4_);}

  //! return the 'row map' covering the ranges of the first two indices i1,i2.
  const TensorMap2D& map12() const{return map12_;}

  //! return the 'column map' covering the ranges of the last two indices i3,i4.
  const TensorMap2D& map34() const{return map34_;}

  //! retrieve a phist map (required for creating phist vectors with the same data distribution).
  phist_const_map_ptr get_phist_map() const {return phist_map_->get();}

  MPI_Comm comm() const {return comm_;}
  int rank() const {return rank_;}
  int nproc() const {return nproc_;}
  int const* counts() const {return counts_.get();}
  int const* disps() const {return disps_.get();}
  gidx offset() const {return offset_;}

  protected:

    int n1_, n2_, n3_, n4_;
    gidx offset_;
    lidx local_len_;
    std::shared_ptr<phist::MapOwner> phist_map_;

    // MPI communicator
    MPI_Comm comm_;

    // my rank and the number of processes in comm_
    int rank_, nproc_;

    // This object can be viewed as a matricization with the first and second,
    // respectively the third and fourth dimension combined to form the row resp.
    // column index space.
    TensorMap2D map12_, map34_;

    // number of local elements on proc i
    std::shared_ptr<int[]> counts_;
    // displacements computed from counts (scan sum over counts)
    std::shared_ptr<int[]> disps_;

};
#endif

