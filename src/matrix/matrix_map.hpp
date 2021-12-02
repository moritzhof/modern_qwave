
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

