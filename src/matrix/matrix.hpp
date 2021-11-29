#pragma once

/*
Written Moritz Hof:

  DLR      - High Performance Computing - Parallel Algorithms and Numerics Group
  TU Delft - Chair of Numerical Analysis,
           - Work for Phd in Applied and Compututional Mathematics

Basic Linear algebra functions: naively parallelized
*/
#include "../coord/coord.hpp"
#include "matrix_map.hpp"


#include <vector>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include <ranges>
#include <execution>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include "mkl_cblas.h"


namespace qwv{

//complete matrix class
// T denotes the element type,
// exec the standard execution policy to be used
// (for loops that require sequential execution this may be overridden)
template<typename T>
class matrix{

private:

 static constexpr auto exec=std::execution::par_unseq;

public:

 matrix() = default;
    
 matrix(std::size_t rows, std::size_t cols) : _row(rows), _col(cols), _map(rows,cols) {
   mat.resize(rows*cols);
   std::for_each(exec, _map.begin(), _map.end(),
        [&](auto idx){auto [i,j]=idx; (*this)(i,j)= static_cast<T>(1.0);});
   }

    
  matrix(std::initializer_list<T> matx) : _row(matx.size()), _col(1), _map(_row, _col) {
    mat.reserve(_row*_col);
    std::for_each(exec,std::begin(matx), std::end(matx), [&](int i){
       mat.emplace_back(i);
     });
  }
    
    matrix(const matrix& A) = default; //:_row{A._row}, _col{A._col}, mat{A.mat} {}
    matrix(matrix&& A) = default; //:_row{A._row}, _col{A._col}, mat{A.mat} {}
    

    matrix(std::initializer_list<std::initializer_list<T>> matx) : _row(matx.size()) ,
       _col(matx.begin()->size()), mat( matx.size() * matx.begin()->size() ) {

       auto v = std::views::iota(static_cast<std::size_t>(0), _row);
       std::for_each(std::execution::par_unseq, std::begin(v), std::end(v), [&](auto i){
        for(int j = 0; j < _col; ++j)
          mat[i*_col+j] = ((matx.begin()+i)->begin())[j] ;
        });
      }

  ~matrix(){}
    
    matrix<T>& operator = (matrix<T> const& rhs){
        matrix<T> copy = rhs;
        std::swap(*this, copy);
        return *this;
    }
    
    matrix<T>& operator = (matrix<T> && rhs){
        std::swap(_row, rhs._row);
        std::swap(_col, rhs._col);
        std::swap(mat, rhs.mat);
        return *this;
    }

    std::size_t numrows() const {return _row;}
    std::size_t numcols() const {return _col;}

 const T& operator[](std::size_t index) const { return mat[index];}
       T& operator[](std::size_t index){return mat[index];}
 const T& operator()(int i, int j) const { return mat[j*_row+i];}
       T& operator()(int i, int j){return mat[j*_row+i];}


 matrix<T> operator*(const matrix<T>& B) const{

    //not this component-wise vector multiplation: NOT dot_product: (for Dot-Product: call method dot)
     if ((B._col == 1) && (_col == 1) && (B._row == _row)){
      matrix<T> res(_row, 1);
      auto v = std::views::iota(static_cast<std::size_t>(0), B.mat.size());
       std::for_each(exec, std::begin(v), std::end(v), [&](auto i){
         res[i] = mat[i]*B.mat[i];
       });
       return res;
    }

    if(B._col == 1 && B._row == _row){
    matrix<T> res(_row, 1);
    auto v = std::views::iota(static_cast<std::size_t>(0), static_cast<std::size_t>(_row));
       std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
             for(int j = 0; j < _col; ++j){
               res[i] += mat[j*_col+i]*B[j];
           }
       });

       return res;
    }

    if (_col != B._row) throw std::runtime_error("in A*B: incompatible dimensions.");
    // TODO: needs adjusting for T!=double
    matrix<T> result(_row, B._col);
    double alpha=1.0, beta=0.0;
    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 static_cast<int>(_row), static_cast<int>(B._col), static_cast<int>(_col),
                 alpha, &mat[0], std::max(1,static_cast<int>(_row)), &B[0], std::max(1, static_cast<int>(B._col)),
                 beta, &result[0], std::max(1, static_cast<int>(result._col))  );

    return result;
  }

    
    matrix<T> operator+(matrix<T>& B){

      assert(_row == B._row && _col == B._col);
      matrix result(B._row, B._col);
      auto v = std::views::iota(static_cast<std::size_t>(0), _row*_col);
     std::for_each(std::execution::par_unseq, std::begin(v), std::end(v), [&](auto i){
      
         for(int j = 0; j < _col; ++j){
            result[i*_col+j] = this->mat[i*_col+j]+B[i*_col+j];
          }
        });

        return result;
    }

    matrix<T> operator-(matrix<T>& B){

      assert(_row == B._row && _col == B._col);
      matrix<T> result(_row, _col);
      auto v = std::views::iota(static_cast<std::size_t>(0), _row*_col);
      std::for_each(std::execution::par_unseq, std::begin(v), std::end(v), [&](auto i){
         result[i] = this->mat[i]-B[i];
    });
        return result;
    }

    


   matrix<T> reshape(std::size_t n, std::size_t m);
   matrix<T> reshape_with_padding(std::size_t n, std::size_t m);
   matrix<T> submat(std::size_t startrow, std::size_t endrow, int startcol, int endcol );

  friend std::ostream& operator<<(std::ostream& os, const matrix& _mat){

    for(int i = 0; i < _mat._row; ++i){
      for(int j = 0; j < _mat._col; ++j){
        os <<  std::setprecision(std::numeric_limits<T>::digits10+1)<< _mat.mat[j*_mat._col+i] << "  "; } os << '\n';
       
    }
    return os;
  }


  std::vector<T> mat;
  std::size_t _row;
  std::size_t _col;

  TensorMap2D _map;

};


// //returns nxm matrix in new from of n' x m'
template<typename T>
matrix<T> matrix<T>::reshape(std::size_t n, std::size_t m){

matrix<T> result(n, m);
  assert(n*m == _row*_col);
  auto v = std::views::iota(static_cast<std::size_t>(0), n*m);
    std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
        result[(i/m)*m + i%m] = std::forward<T>(mat[(i/_col)*_col+i % _col]);
      });

  return result;
}

// //will pad with zeros when desried reshape bigger is: nothing special. for smaller,
// // use submat
template<typename T>
matrix<T> matrix<T>::reshape_with_padding(std::size_t n, std::size_t m){

  matrix<T> result(n, m);
  auto v = std::views::iota(static_cast<std::size_t>(0), _row);
    std::for_each(std::execution::par, std::begin(v), std::end(v), [&](auto i){
      for(int j = 0; j < _col; ++j)
        result[i*m+j] = std::forward<T>(mat[i*_col+j]);
      });

  return result;
}

template<typename T>
matrix<T> matrix<T>::submat(std::size_t startrow, std::size_t endrow, int startcol, int endcol ){
  auto row_length = endrow - startrow;
  auto col_length = endcol - startcol;

  matrix<T> submatrix(row_length+1, col_length+1);
  assert(submatrix._row <= _row && submatrix._col <= _col);
  for(int i = startrow; i <= endrow; ++i){
    for(int j = startcol; j <= endcol; ++j){
    submatrix[(i-startrow)*submatrix._col+(j-startcol)] = std::forward<T>(mat[i*_col+j]);
  }
}
  return submatrix;
}


} // end of namespace qwv


/*
 
 matrix(std::initializer_list<std::initializer_list<T>> matx) : _row(matx.size()) ,
       _col(matx.begin()->size()), _map(matx.size(), matx.begin()->size()) {

        mat.reserve(_row*_col);
        std::for_each(exec, _map.begin(), _map.end(), [&](auto iter){
            auto[i,j] = iter;
            mat[i*_col+j] = ((matx.begin()+i)->begin())[j];
            std::cout << "djsfhsadkljhfkjsadhfkjsadh" << std::endl;
        });

   }

 
   matrix<T> operator+(matrix<T> const& B){
 
     assert(_row == B._row && _col == B._col);
     matrix<T> result(B._row, B._col);
     std::for_each(exec, _map.begin(),_map.end(), [&, this](auto const idx){
           auto [i,j]=idx;
           //result(i,j) = (*this)(i,j) + B(i,j);
         result(i,j) = mat[i*_col+j] + B(i,j);
       });
       std::cout << "OPERATOR CALLED\n";
       return result;
   }
 
   matrix<T> operator-(matrix<T>& B){
 
     assert(_row == B._row && _col == B._col);
     matrix<T> result(_row, _col);
     std::for_each(exec,_map.begin(),_map.end(), [&](auto idx){
           auto [i,j]=idx;
           result(i,j) = (*this)(i,j)-B(i,j);
       });
       return result;
   }
 
 */
