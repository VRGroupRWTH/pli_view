cmake_minimum_required(VERSION 3.0.2)

# Setup project metadata.
set(ProjectName "pli_vis")
project(${ProjectName} VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -fopenmp -lgomp")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda" "-gencode=arch=compute_35,code=sm_35" "-gencode=arch=compute_50,code=sm_50" "-gencode=arch=compute_61,code=sm_61")

# Setup dependencies.
include("cmake/_dependencies.cmake")

# Setup sources.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/third_party)
include("cmake/_sources.cmake")

# Create source groups.
include("cmake/assign_source_group.cmake")
assign_source_group(${ProjectSources})

# Specify output.
cuda_add_executable(${ProjectName} ${ProjectSources})

# Link libraries.
target_link_libraries(${ProjectName} ${ProjectLibraries})
