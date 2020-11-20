cmake_minimum_required(VERSION 3.17)
project(cuda CUDA)
#set(CMAKE_CUDA_FLAGS "-arch=sm_20")
set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(cuda src/main.cu)
set_target_properties(
        cuda
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex1_ex1 src/Ex1/Ex1.cu)
set_target_properties(
        ex1_ex1
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex2_err1 src/Ex2/err1.cu)
set_target_properties(
        ex2_err1
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex2_err2 src/Ex2/err2.cu)
set_target_properties(
        ex2_err2
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex3 src/Ex3/Ex3.cu)
set_target_properties(
        ex3
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex4 src/Ex4/reduce.cu)
set_target_properties(
        ex4
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        CUDA_SEPARABLE_COMPILATION ON)
