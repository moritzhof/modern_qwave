
#pragma once 

#include "../qwv_types.hpp"

#include<tuple>
#include<ranges>

namespace qwv {


class coord : public std::iterator<std::input_iterator_tag, std::tuple<int, int>>{
public:
    using pos_type = uint32_t;
    
    coord() : pos(0), n1_(0), n2_(0) {
        std::tie(n1_, n2_);
    }
    
    coord(std::tuple<int, int> i, std::tuple<int, int> n) : n1_(std::get<0>(n)), n2_(std::get<1>(n)), pos(std::get<1>(i)*std::get<0>(n)+std::get<0>(i)) {}
    
    explicit coord(pos_type pos_in, std::tuple<int, int> n) : n1_(std::get<0>(n)), n2_(std::get<1>(n)) {}
    
    coord& operator++(){++pos; return *this; }
    coord operator++(int){ return coord(++pos, std::make_tuple(n1_,n2_)); }
   // coord operator++(int){ coord result(pos, std::tie(n1_, n2_)); result.pos++; return result;  }
    std::tuple<int, int> operator*() const{
        return this->operator[](0);
    }
    
    std::tuple<int, int> operator[](pos_type n) const{
        pos_type pos_n = pos + n;
        if(pos_n < 0 || pos_n >= n1_*n2_){ throw std::runtime_error("iteration out-of-bound"); }
        int i1=pos_n%n1_;
        int i2=pos_n/n1_;
        return std::make_tuple(i1,i2);
    }
    
    template<typename int_type>
    coord operator+(int_type k) const{
        return coord(pos+k, std::make_tuple(n1_, n2_));
    }
    
    template<typename int_type>
    coord operator+(coord const& other){
        if(n1_ != other.n1_ || n2_ != other.n2_) throw std::logic_error("dimension mismatch in iterator arithmetic");
        return coord(pos+other.pos, std::make_tuple(n1_, n2_));
    }
    
    template<typename int_type>
    coord operator-(int_type k) const{
        return coord(pos-k, std::make_tuple(n1_, n2_));
    }
    
    template<typename int_type>
    coord operator-(coord const& other){
        if(n1_ != other.n1_ || n2_ != other.n2_) throw std::logic_error("dimension mismatch in iterator arithmetic");
        return coord(pos-other.pos, std::make_tuple(n1_, n2_));
    }
    
    coord& operator+=(std::make_signed<pos_type>::type n) { pos += n; return *this; }
    coord& operator-=(std::make_signed<pos_type>::type n) { pos -= n; return *this; }
   
    bool operator==(coord const& other){
        bool equal = ((n1_ == other.n1_) && (n2_ == other.n2_ ));
        return equal;
    }
    
    bool operator!=(coord const& other) {return !(*this == other); }
    
    

    pos_type pos;
private:
    const int n1_, n2_;
    
    
};

class MultiIndex4D: public std::iterator<std::input_iterator_tag, std::tuple<int,int,int,int>>
{
public:
  using pos_type=gidx;
  using quad=std::tuple<int,int,int,int>;

  MultiIndex4D(quad const& i, quad const& n)
    : n1_(std::get<0>(n)),
      n2_(std::get<1>(n)),
      n3_(std::get<2>(n)),
      n4_(std::get<3>(n))
  {
    const pos_type i1=std::get<0>(i);
    const pos_type i2=std::get<1>(i);
    const pos_type i3=std::get<2>(i);
    const pos_type i4=std::get<3>(i);
    pos = ((i4*n3_ + i3)*n2_+i2)*n1_+i1;
  }

  MultiIndex4D(pos_type pos_in, quad const& n) : MultiIndex4D(quad{0,0,0,0},n) {pos=pos_in;}

  MultiIndex4D& operator++(){++pos; return *this;}

  MultiIndex4D operator++(int){return MultiIndex4D(pos++,std::tie(n1_,n2_,n3_,n4_));}

  quad operator*() const {return this->operator[](0);}

  quad operator[](lidx n) const
  {
    lidx pos_n=pos+n;
    if (pos_n<0||pos_n>=end_pos())
    {
      throw std::runtime_error("iterator out-of-bounds");
    }

    lidx rem = pos_n;
    int i1 = rem % n1_;
    rem = rem / n1_;
    int i2 = rem % n2_;
    rem = rem / n2_;
    int i3 = rem % n3_;
    rem = rem / n3_;
    int i4 = rem % n4_;
    return std::tie(i1,i2,i3,i4);
  }

  //! return an iterator advanced by k steps
  template<typename int_type>
  MultiIndex4D operator+(int_type k) const
  {
    return MultiIndex4D(pos+static_cast<pos_type>(k),std::tuple{n1_,n2_,n3_,n4_});
  }

  bool operator!=(const MultiIndex4D& other)
  {
    return n1_!=other.n1_ || n2_!=other.n2_||n3_!=other.n3_||n4_!=other.n4_||pos!=other.pos;
  }

  //! absolute position of the iterator in a 1D array
  pos_type pos;

  private:
    const int n1_, n2_, n3_, n4_;
    constexpr pos_type end_pos() const {return pos_type(n1_)*pos_type(n2_)*pos_type(n3_)*pos_type(n4_);}
};


}
