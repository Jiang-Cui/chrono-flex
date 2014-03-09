# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(ImplicitBeamsGPU)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################

# include CUSP
SET(CUSP_INC_DIR "" CACHE PATH "Where is the CUSP include directory?")
include_directories(${CUSP_INC_DIR})

# include SPIKE
SET(SPIKE_INC_DIR "" CACHE PATH "Where is the SPIKE include directory?")
include_directories(${SPIKE_INC_DIR})

## include SSCD
#SET(SSCD_INC_DIR "" CACHE PATH "Where is the SSCD include directory?")
#include_directories(${SSCD_INC_DIR})
#SET(SSCD_LIB_PATH       ""   CACHE PATH "Where is SSCD located")		 
#FIND_LIBRARY( SSCD_LIB NAMES sscd PATHS ${SSCD_LIB_PATH})

## Example MPI program
#enable_mpi_support()
#add_executable(integrate_mpi integrate_mpi.cpp)
#target_link_libraries(integrate_mpi ${MPI_CXX_LIBRARIES})

# Example OpenMP program
enable_openmp_support()
#add_executable(integrate_omp integrate_omp.cpp)


IF (CUDA_SDK_ROOT_DIR)
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
  ADD_DEFINITIONS(-DWITH_CUDA_SDK)
ENDIF

## Using Armadillo
#find_package( Armadillo REQUIRED )
#include_directories(${ARMADILLO_INCLUDE_DIRS})

# include OpenGL
find_package("GLUT")
find_package("OpenGL")
#freeglut is not found properly on windows
IF ("${CH_OS}" STREQUAL "WINDOWS")
	SET(FREEGLUT_LIB_PATH       ""    	CACHE PATH "Where is freeglut located?")		 	
	FIND_LIBRARY( FREEGLUT_LIB NAMES glut glut32 freeglut PATHS ${FREEGLUT_LIB_PATH})
ENDIF()
MARK_AS_ADVANCED(FORCE FREEGLUT_LIB)

# If GLUT was found, define the macro WITH_GLUT
IF (${GLUT_glut_LIBRARY})
	ADD_DEFINITIONS(-DWITH_GLUT)
ENDIF()

# Example CUDA program
enable_cuda_support()

# Create the implicit-beams-GPU library
cuda_add_library(ibeamsgpuLIB SHARED ANCFSystem.cuh ANCFSystem.cu ANCFintForce.cu include.cuh ANCFConstraints.cu)# ANCFContact.cu)
target_link_libraries(ibeamsgpuLIB 
	${GLUT_LIBRARIES}
	${FREEGLUT_LIB}
	${OPENGL_LIBRARIES}
#	${ARMADILLO_LIBRARIES}
#	${SSCD_LIB}
	)
set_target_properties(ibeamsgpuLIB PROPERTIES COMPILE_DEFINITIONS "IBEAMS_COMPILE_LIBRARY")

# Create the executable
cuda_add_executable(ImplicitBeamsGPU main.cu)
target_link_libraries(ImplicitBeamsGPU ibeamsgpuLIB)
add_dependencies(ImplicitBeamsGPU ibeamsgpuLIB)
