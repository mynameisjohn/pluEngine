# The CMakeLists.txt for Link3Dt

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

file(GLOB SOURCES Source/*.cpp)
file(GLOB HEADERS Include/*.h)

add_library(Link3Dt ${SOURCES} ${HEADERS})

target_include_directories(Link3Dt Include)
