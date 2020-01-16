###############################################################################
###############################################################################
#
# Include directory for sources
#
include_directories(${DIFFUSION_INCLUDE_DIR})
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Name all sources
#
#file(GLOB_RECURSE DIFFUSION_TARGET_LIB_SRC  "coefficients/*.cc") # source files
#set(DIFFUSION_TARGET_LIB_SRC ${DIFFUSION_TARGET_LIB_SRC})

set(DIFFUSION_COEFFICIENTS_LIB_SRC
	coefficients/basis_q1.inst.cc
	coefficients/dirichlet_bc.inst.cc
	coefficients/matrix_coeff.inst.cc
	coefficients/neumann_bc.inst.cc
	coefficients/right_hand_side.inst.cc)

set(DIFFUSION_LIB_SRC
	base/diffusion_problem.inst.cc
	base/diffusion_problem_basis.inst.cc
	base/diffusion_problem_ms.inst.cc)

print_all_args (
  ${DIFFUSION_COEFFICIENTS_LIB_SRC}
 )

print_all_args (
  ${DIFFUSION_LIB_SRC}
 )
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#
add_library (DIFFUSION_COEFFICIENTS_LIBRARY SHARED ${DIFFUSION_COEFFICIENTS_LIB_SRC})
DEAL_II_SETUP_TARGET(DIFFUSION_COEFFICIENTS_LIBRARY)

add_library (DIFFUSION_LIBRARY SHARED ${DIFFUSION_LIB_SRC})
DEAL_II_SETUP_TARGET(DIFFUSION_LIBRARY)

#
# Compile the executables
#
add_executable(DIFFUSION "main.cxx")
DEAL_II_SETUP_TARGET(DIFFUSION)
TARGET_LINK_LIBRARIES(DIFFUSION DIFFUSION_LIBRARY
							DIFFUSION_COEFFICIENTS_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${DIFFUSION_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})
###############################################################################
###############################################################################
