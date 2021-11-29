//
//  qwv_util.hpp
//  
//
//  Created by Moritz Hof on 22.11.21.
//

#pragma once

#include <stdio.h>
#include<iostream>
#include<ranges>
#include<execution>
#include<algorithm>
#include<cmath>
#include<span>

namespace qwv{
  namespace util{
  
    auto to_vector(auto&& range){
      std::vector<std::ranges::range_value_t<decltype(range)>> v;
      if constexpr (std::ranges::sized_range<decltype(range)>){
        v.reserve(std::ranges::size(range));
      }

       std::ranges::copy(range,std::back_inserter(v));
    return v;
  }




  template <typename F, typename G>
  constexpr decltype(auto) operator|(F &&f, G &&g) {
    if constexpr (std::is_invocable_v<G &&, F &&>)
      return std::forward<G>(g)(std::forward<F>(f));
   else
     return [f = std::forward<F>(f), g = std::forward<G>(g)](auto &&... args) {
       return g(f(std::forward<decltype(args)>(args)...));
   };
  }
  
  template<typename T>
  void print_vector(std::vector<T> const& vec){
      for(const auto & i : vec){ std::cout << i << ' '; }
      std::cout << '\n';
  }
  
  
 } //end of util namespace
} //end of qwv namespace
 /* qwv_util_hpp */
