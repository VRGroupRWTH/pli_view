cmake_minimum_required(VERSION 3.0)

# Setup project metadata.
set(ProjectName "pli_vis")
project(${ProjectName} VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Setup dependencies.
include("cmake/_dependencies.cmake")

# Setup sources.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include("cmake/_sources.cmake")

# Create source groups.
include("cmake/assign_source_group.cmake")
assign_source_group(${ProjectSources})

# Specify output.
add_executable(${ProjectName} ${ProjectSources})

# Link libraries.
target_link_libraries(${ProjectName} ${ProjectLibraries})
