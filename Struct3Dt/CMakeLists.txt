# The CMakeLists.txt for Struct3Dt

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

file(GLOB SOURCES Source/*.cpp)
file(GLOB HEADERS Include/*.h)

add_library(Struct3Dt ${SOURCES} ${HEADERS})

target_include_directories(Struct3Dt Include)
