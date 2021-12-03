
![qwave banner](./images/qwaves_banner.png)
## Table of Contets
* [General Information](#general-information)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)

## General Information
Qwaves is a quantum simulations to for higher-dimensional three body problems

## Dependencies
* C++20
* Phist with trilonis
* cuBLAS or CUTLASS

## Setup
```
$ cmake ..
```

# Usage
```cpp
auto chebyshev = qwv::differential::Chebyshev1D(range, N);
```
