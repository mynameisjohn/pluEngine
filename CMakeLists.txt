# The CMakeLists.txt file for PLuTARC
# This includes CenterFind2D, Link3Dt, and Struct3Dt 
# as subdirectories and builds them all.
# The main program then runs those three as phases on
# a set of input data (a TIFF stack.) 

# TODO
# Set up OpenCV directories
# Get hdf5 and FreeImage set up
# Keep IPP around for tests and verification
# Learn about Link3Dt and Struct3Dt
# Write GPU kernels for the statistical phase of CenterFind2D
# as well as whatever Link3Dt and Struct3Dt do. 

cmake_minimum_required(VERSION 2.8)

project(PLuTARC)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

add_subdirectory(CenterFind2D)
add_subdirectory(Link3Dt)
add_subdirectory(Struct3Dt)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/CenterFind2D/Include
					${CMAKE_CURRENT_SOURCE_DIR}/Struct3Dt/Include
					${CMAKE_CURRENT_SOURCE_DIR}/Link3Dt/Include)

set(SOURCES main.cpp)

add_executable(PLuTARC_Tests ${SOURCES})

target_link_libraries(PLuTARC_Tests LINK_PUBLIC 
									CenterFind2D 
									Link3Dt 
									Struct3Dt)

