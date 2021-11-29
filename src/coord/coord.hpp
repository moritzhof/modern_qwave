
#pragma once 

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
}
