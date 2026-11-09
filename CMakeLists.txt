cmake_minimum_required(VERSION 3.18)
project(ParallelDecisionTree LANGUAGES CXX)

option(DT_ENABLE_CUDA "Build the CUDA histogram backend" OFF)
option(DT_NATIVE "Optimise for the build machine (-march=native)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(DT_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(DT_CORE_SOURCES
  src/dataset.cpp
  src/binning.cpp
  src/split.cpp
  src/tree.cpp)

# Core without OpenMP, used by the serial reference and the unit tests.
add_library(dt_core_serial STATIC ${DT_CORE_SOURCES})
target_include_directories(dt_core_serial PUBLIC include)
target_compile_options(dt_core_serial PUBLIC -Wno-unknown-pragmas)

# Core with OpenMP plus the CPU histogram backend, used by the MPI program.
add_library(dt_core STATIC ${DT_CORE_SOURCES} src/hist_cpu.cpp)
target_include_directories(dt_core PUBLIC include)
target_link_libraries(dt_core PUBLIC OpenMP::OpenMP_CXX)

add_executable(dt_serial src/main_serial.cpp)
target_link_libraries(dt_serial PRIVATE dt_core_serial)

add_executable(dt_mpi src/main_mpi.cpp src/trainer.cpp)
target_link_libraries(dt_mpi PRIVATE dt_core MPI::MPI_CXX)

if(DT_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(dt_mpi PRIVATE src/hist_cuda.cu)
  target_compile_definitions(dt_mpi PRIVATE DT_HAVE_CUDA)
  target_link_libraries(dt_mpi PRIVATE CUDA::cudart)
  set_target_properties(dt_mpi PROPERTIES CUDA_STANDARD 17)
endif()

add_executable(dt_tests tests/test_core.cpp)
target_link_libraries(dt_tests PRIVATE dt_core_serial)

enable_testing()
add_test(NAME core COMMAND dt_tests)
