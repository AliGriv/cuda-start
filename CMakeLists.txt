cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0" )
set(CMAKE_CUDA_STANDARD 14)
message("hooo ")
message("include cuda: ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
project(cuda_start LANGUAGES CXX CUDA)
enable_language(CUDA)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")
add_library(funcs STATIC funcs.h)
set_target_properties( funcs PROPERTIES LINKER_LANGUAGE CXX)
add_library(funcs_cuda STATIC funcs_cuda.cuh funcs_cuda.cu)
target_include_directories(funcs_cuda PUBLIC ${CMAKE_SOURCE_DIR}/inc)
set_target_properties( funcs_cuda PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(funcs_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




add_executable(test_cuda_start main.cpp)
#set_target_properties(test_cuda_start PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_cuda_start PRIVATE funcs funcs_cuda)