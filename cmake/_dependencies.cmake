# Register bundled modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Include Cuda and Cublas.
find_package(CUDA REQUIRED)
set(ProjectLibraries ${ProjectLibraries} "${CUDA_CUBLAS_LIBRARIES};${CUDA_cusolver_LIBRARY};${CUDA_cudadevrt_LIBRARY}")

# Include HDF5.
set(HDF5_INCLUDE_DIR CACHE STRING "")
set(HDF5_C_SHARED_LIBRARY CACHE STRING "")
find_package(HDF5 NAMES hdf5 COMPONENTS C shared)
if(HDF5_INCLUDE_DIR STREQUAL "" OR HDF5_C_SHARED_LIBRARY STREQUAL "")
  message(FATAL_ERROR "HDF5 not found! Please set the include directory and libraries manually.")
elseif()
  include_directories(${HDF5_INCLUDE_DIR})
  set(ProjectLibraries ${ProjectLibraries} ${HDF5_C_SHARED_LIBRARY})
endif()

# Include OpenGL.
find_package       (OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
set(ProjectLibraries ${ProjectLibraries} "${OPENGL_LIBRARIES}")

# Include Qt.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Widgets REQUIRED)
set(ProjectLibraries ${ProjectLibraries} Qt5::Widgets)
