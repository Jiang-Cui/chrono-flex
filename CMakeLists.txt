#=============================================================================
# CMake configuration file for ImplicitBeamsGPU
#=============================================================================

# Minimum version of CMake required. Don't touch.
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Set the name of your project
PROJECT(ImplicitBeamsGPU)

# Include macros from the SBEL utils library
INCLUDE(SBELUtils.cmake)

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

# ----------------------------------------------------------------------------
# Require CUSP (headers-only library)

SET(CUSP_INC_DIR "" CACHE PATH "Where is the CUSP include directory?")
INCLUDE_DIRECTORIES(${CUSP_INC_DIR})

# ----------------------------------------------------------------------------
# Require SPIKE (headers-only library)

SET(SPIKE_INC_DIR "" CACHE PATH "Where is the SPIKE include directory?")
INCLUDE_DIRECTORIES(${SPIKE_INC_DIR})

# ----------------------------------------------------------------------------
# Require OpenMP (?)

enable_openmp_support()

# ----------------------------------------------------------------------------
# If the CUDA_SDK_ROOT_DIR is defined

IF (CUDA_SDK_ROOT_DIR)
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
  ADD_DEFINITIONS(-DWITH_CUDA_SDK)
ENDIF()

# ----------------------------------------------------------------------------
# Using Armadillo

#FIND_PACKAGE( Armadillo REQUIRED )
#INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})

# ----------------------------------------------------------------------------
# Include OpenGL support

FIND_PACKAGE("GLUT")
FIND_PACKAGE("OpenGL")

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

# ----------------------------------------------------------------------------
# Enable CUDA support

enable_cuda_support()

# ----------------------------------------------------------------------------
# List the files that make up the ImplicitBeamsGPU library

SET(IBEAMS_SOURCES
	src/ANCFApi.h
	src/include.cuh
	src/ANCFSystem.cuh
	src/ANCFSystem.cu
	src/ANCFConstraints.cu
	src/ANCFintForce.cu
	#src/ANCFContact.cu
	src/Constraint.cuh
	src/Element.cuh
	src/Node.cuh
	src/Particle.cuh
	)

SOURCE_GROUP(src FILES ${IBEAMS_SOURCES})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

# ----------------------------------------------------------------------------
# Add the ImplicitBeamsGPU library

CUDA_ADD_LIBRARY(ImplicitBeamsGPU SHARED ${IBEAMS_SOURCES})
TARGET_LINK_LIBRARIES(ImplicitBeamsGPU 
	${GLUT_LIBRARIES}
	${FREEGLUT_LIB}
	${OPENGL_LIBRARIES}
#	${ARMADILLO_LIBRARIES}
	)
SET_TARGET_PROPERTIES(ImplicitBeamsGPU PROPERTIES COMPILE_DEFINITIONS "IBEAMS_COMPILE_LIBRARY")

INSTALL(TARGETS ImplicitBeamsGPU
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

# ----------------------------------------------------------------------------
# Add the models

ADD_SUBDIRECTORY(models)

