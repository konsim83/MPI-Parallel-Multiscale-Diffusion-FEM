#############################################################
#############################################################
# Print all arguments
macro (print_all_args)
  message (STATUS "------------------------------------")
  
  set(list_var "${ARGN}")
  foreach(_currentItem IN LISTS list_var)
    message (STATUS "Adding library source file:   ${_currentItem}")
  endforeach (_currentItem)
  
  message (STATUS "------------------------------------")
endmacro (print_all_args)
#############################################################
#############################################################



###############################################################################
###############################################################################
message(STATUS "This is CMake ${CMAKE_VERSION}")
message(STATUS "")

cmake_minimum_required(VERSION 2.8.12)

find_package(deal.II 9.1.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if (NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug."
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release."
  )
  
ADD_CUSTOM_TARGET(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./contrib/utilities/indent
  COMMENT "Indenting all C++ source and header files."
  ) 

set(PROJECT_NAME DIFFUSION)

# The version number
set(DIFFUSION_VER_MAJOR 0)
set(DIFFUSION_VER_MINOR 1)

project(${PROJECT_NAME})
###############################################################################
###############################################################################



###############################################################################
###############################################################################
# Check for the existence of various optional folders:
if (EXISTS ${CMAKE_SOURCE_DIR}/doc/CMakeLists.txt)
  set(DIFFUSION_HAVE_DOC_DIRECTORY TRUE)
endif ()

if (EXISTS ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt)
  set (DIFFUSION_HAVE_TEST_DIRECTORY TRUE)
endif ()
###############################################################################
###############################################################################



# Change default CMAKE_INSTAL_PREFIX to ${CMAKE_BINARY_DIR}/lib
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/lib" CACHE PATH "default install path" FORCE )
endif ()

set (DIFFUSION_LIBRARY DIFFUSION)


###############################################################################
###############################################################################
set(DIFFUSION_INCLUDE_DIR
		${CMAKE_SOURCE_DIR}/include)

set(DIFFUSION_SRC_DIR 
		${CMAKE_SOURCE_DIR}/source)
add_subdirectory (${DIFFUSION_SRC_DIR})


# Only add subdirectories for doc if exists
if (DIFFUSION_HAVE_DOC_DIRECTORY)
  add_subdirectory (${CMAKE_SOURCE_DIR}/doc)
endif ()

# Only add subdirectories for doc if exists
if (DIFFUSION_HAVE_TEST_DIRECTORY)
  add_subdirectory (${CMAKE_SOURCE_DIR}/test)
  ENABLE_TESTING()
endif ()
###############################################################################
###############################################################################