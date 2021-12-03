
![qwave banner](./images/qwaves_banner.png)
## Table of Contets
* [General Information](#general-information)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)
* [Developers](#developers)

## General Information
Qwaves is a quantum simulations to for higher-dimensional three body problems

## Dependencies
* C++20
* Phist with Trilinos
* cuBLAS or CUTLASS

## Setup
```
$ cmake ..
```

## Usage
```cpp
std::size_t N = ... ;
auto roots = qwv::discretization::roots<double>{N}
auto chebyshev = qwv::differential::Chebyshev1D<double>(roots, N);

// CUDA

```
## Developers
* Moritz Travis Hof:
     - German Aerospace Agency: High Performance Computing : Parallel Algorithms and Numerics
     - Technical University of Delft - Department of Applied Mathematics
     
* Dr. Jonas Thies:
     - Technical University of Delft - Department of Applied Mathematics
