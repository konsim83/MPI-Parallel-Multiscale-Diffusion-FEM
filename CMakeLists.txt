#
# General configuration for cmake:
#
SET(PROJECT_NAME ELLIPTIC_MS_MPI)

#
# The version number.
#
SET(ELLIPTIC_MS_MPI_VER_MAJOR 0)
SET(ELLIPTIC_MS_MPI_VER_MINOR 1)


#
# Check for the existence of various optional folders:
#
if (EXISTS ${CMAKE_SOURCE_DIR}/doc/CMakeLists.txt)
  set (ELLIPTIC_MS_MPI_HAVE_DOC_DIRECTORY TRUE)
endif ()

if (EXISTS ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt)
  set (ELLIPTIC_MS_MPI_HAVE_TEST_DIRECTORY TRUE)
endif ()


# Set the name of the project and target:
SET(TARGET "main")

SET(INCLUDE_DIR
		${CMAKE_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${INCLUDE_DIR})

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

SET(SRC_DIR 
		${CMAKE_SOURCE_DIR}/source)

#SET(TARGET_SRC
#  ${SRC_DIR}/${TARGET}.cc
#  )
FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")


###############################################################################
###############################################################################
# Usually, you will not need to modify anything beyond this point...
###############################################################################
###############################################################################

MESSAGE(STATUS "This is CMake ${CMAKE_VERSION}")
MESSAGE(STATUS "")

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.1.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${PROJECT_NAME})
DEAL_II_INVOKE_AUTOPILOT()
