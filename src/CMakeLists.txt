
#set(C_LIBSOURCE  buildMatrix.c SparseMatrixOperations.c pSparseMatrixOperations.c #StandardMatrixOperations.c Statistics.c output.c)

set(CXX_LIBSOURCE) # buildOperator.cpp buildPreconditioner.cpp TensorMap.cpp)

#if (ENABLE_PARPACK)
#  list(APPEND C_LIBSOURCE parpack_interface.c)
#  list(APPEND CXX_LIBSOURCE parpack_interface.cpp)
#endif()

if (HAVE_PHIST)
  add_definitions(-DHAVE_PHIST=1)
  list(APPEND CXX_LIBSOURCE phist_interface.hpp ddgemm.hpp)
endif()

add_library(tbbs STATIC ${C_LIBSOURCE})
add_library(tbbs_cxx STATIC ${CXX_LIBSOURCE})

#add_executable(three_body_bound_state 1D_problem_DV_hp.c)
#target_link_libraries(three_body_bound_state tbbs quadmath)

#if (ENABLE_PARPACK)
#  add_executable(three_body_bound_state_cxx 1D_problem_DV_hp.cpp)
#  target_link_libraries(three_body_bound_state_cxx tbbs_cxx tbbs quadmath)
#endif()

if (HAVE_PHIST)
  add_executable(three_body_bound_state_phist_cxx 1D_problem_DV_hp.cpp)
  target_compile_definitions(three_body_bound_state_phist_cxx PUBLIC -DUSE_PHIST=1)
  target_link_libraries(three_body_bound_state_phist_cxx tbbs_cxx tbbs quadmath)
  target_link_libraries(three_body_bound_state_phist_cxx ${PHIST_LIBRARIES})

  add_executable(2Dthree_body_bound_state_phist_cxx 2D_problem_DV_hp.cpp)
  target_compile_definitions(2Dthree_body_bound_state_phist_cxx PUBLIC -DUSE_PHIST=1)
  target_link_libraries(2Dthree_body_bound_state_phist_cxx tbbs_cxx tbbs quadmath)
  target_link_libraries(2Dthree_body_bound_state_phist_cxx ${PHIST_LIBRARIES})

endif()
