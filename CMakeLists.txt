###
### Minimal cmake version
###
cmake_minimum_required(VERSION 3.16.0)


###
### General project settings
###
### note: the Fortran is for building PARPACK,
### the C++ for GoogleTest
project(qwaves LANGUAGES C CXX Fortran)
set(CMAKE_C_STANDARD_REQUIRED On)
set(CMAKE_C_STANDARD 99)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


###
### Use our own cmake modules
###
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# optionally build with parpack support. Note that parpack doesn't compile
# with gfortran 10. Enabling it will also install it from scratch right now.
option(ENABLE_PARPACK "build and install parpack, and support it as eigensolver." OFF)

###
### Compiler recognition and flags
##
  set(CMAKE_CXX_FLAGS "-march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -lquadmath -ffast-math")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_C_FLAGS "-march=native -Wall -Wextra -Wno-unused -Wno-sign-compare")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -march=native")
  set(CMAKE_C_FLAGS_DEBUG "-O0 -g -march=native")
  message("CMAKE_CXX_FLAGS are ${CMAKE_CXX_FLAGS}")
###
### Required libraries
###
# OpenMP
#find_package(OpenMP REQUIRED)
#link_libraries(OpenMP::OpenMP_C)

find_package(MPI REQUIRED)
link_libraries(MPI::MPI_C MPI::MPI_Fortran)

# BLAS/LAPACK
find_package(BLAS REQUIRED)
link_libraries(${BLAS_LIBRARIES})
find_package(LAPACK REQUIRED)
link_libraries(${LAPACK_LIBRARIES})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMKL_DIRECT_CALL")

if (ENABLE_PARPACK)
  include(PARPACK)
endif()

find_package(phist CONFIG REQUIRED)
if (phist_FOUND)
  message(STATUS "found phist")
  set(HAVE_PHIST ON CACHE BOOL "alternative eigensolvers from phist can be used.")
  include_directories(${PHIST_INCLUDE_DIRS})
else()
  message(STATUS "phist not found, resorting to parpack for eigenvalue problems.")
endif()

###
### Define source files and executables
###
add_subdirectory(src)

configure_file(${PROJECT_SOURCE_DIR}/tbbs_test.cmd.in ${PROJECT_BINARY_DIR}/tbbs_test.cmd @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/tbbs.cmd.in ${PROJECT_BINARY_DIR}/tbbs.cmd @ONLY)

###
### Define test files
###
add_subdirectory(test)
