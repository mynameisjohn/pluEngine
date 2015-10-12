# The CMakeLists.txt for CenterFind2D
# This depends on OpenCV and FreeImage (I don't think
# we need HDF5 till a later phase.)
# It used to rely on IPP, so I may keep that around (probably not.)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

file(GLOB SOURCES Source/*.cpp)
file(GLOB HEADERS Include/*.h)

# You need OpenCV and FreeImage

add_library(CenterFind2D ${SOURCES} ${HEADERS})

target_include_directories(CenterFind2D Include)
