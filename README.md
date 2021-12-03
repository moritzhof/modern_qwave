
![qwave banner](./images/qwaves_banner.png)
## Table of Contets
* [General Information](#general-information)
* [Dependencies](#dependencies)
* [Optional](#optional)
* [Setup](#setup)
* [Usage](#usage)
* [Developers](#developers)

## General Information
Qwaves++ is a computationally and numerically efficient modern templated C++20 approach to determine binding energies and corresponding wave functions of a quantum-mechanical three-body problem in low dimensions. Our approach exploits the tensor structure intrinsic to the multidimensional stationary Schr\"odinger equation, which we express as a discretized eigenvalue problem. In order to obtain numerical solutions of the three-body system in one spatial dimension, we represent the Hamiltonian operator as a series of dense matrix-matrix products and propose an efficient preconditioned Jacobi-Davidson QR iteration for the resulting algebraic eigenvalue problem. This implementation allows a significantly faster computation of three-body bound states with a higher accuracy than in previous works. To investigate the feasibility of solving higher dimensional problems, we also consider cases where we make use of an efficient shared-memory parallel implementation. Our novel approach is of high relevance for investigating the universal behavior of few-body problems governed only by the particle masses, overall symmetries, and the dimensionality of space. Moreover, our results have straightforward applications in the physics of ultracold atomic gases that are nowadays widely utilized as quantum sensors.

## Dependencies
* C++20
* MPI
* Math Kernel Library (MKL)
* Phist with Trilinos (see https://bitbucket.org/essex/phist/wiki/Home )

#### Optional
* NVCC or NVC++
* cuBLAS or CUTLASS

## Setup
```
$ cmake ..
```

### Running with Phist

It is important, after building qwaves++, to copy the jadaOpts file into the build folder you made for qwaves++, otherwise, the proper eigenvalues will not be found.
```
 cp jadaOpts-tbbs.txt build/
```
## Usage
```cpp
//qwaves++ matrix
std::size_t rows = ... , col = ... ;
qwv::matrix<double> mat(rows , cols);
std::size_t N = ... ;
auto roots = qwv::discretization::roots<double>{N}
auto chebyshev = qwv::differential::Chebyshev1D<double>(roots, N);

// CUDA: host: g++11.1 or device: with Nvidia nvc++ compiler with -stdpar flag
auto roots = qwv::discretization::roots<double>{N}
auto chebyshev = qwv::differential::parallel::Chebyshev1D<double>(roots, N);

//CUDA NVCC
auto roots = qwv::cuda::device_memory<double>::allocate_vector(N);
auto chehyshev = qwv::cuda::device_memory<double>::allocate_matrix(N);
qwv::differential::cuda::Chebyshev1D<double><<<blocks,threads>>>(cheby1D, roots, N );
qwv::cuda::synchronize();
```
## Developers
* Moritz Travis Hof:
     - German Aerospace Agency: High Performance Computing - Parallel Algorithms and Numerics
     - Technical University of Delft - Department of Applied Mathematics
     
* Dr. Jonas Thies:
     - Technical University of Delft - Department of Applied Mathematics
