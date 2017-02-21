# Register bundled modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Include Cuda.
find_package(Cuda REQUIRED)
set(ProjectLibraries ${ProjectLibraries} "${CUDA_CUBLAS_LIBRARIES};${CUDA_cudadevrt_LIBRARY}")

# Include Cush.
find_package       (Cush REQUIRED)
include_directories(${CUSH_INCLUDE_DIRS})

# Include GLP.
find_package       (GLP REQUIRED)
include_directories(${GLP_INCLUDE_DIRS})

# Include PLI_IO.
find_package       (PLI_IO REQUIRED)
include_directories(${PLI_IO_INCLUDE_DIRS})

# Include Qt.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Widgets REQUIRED)
set(ProjectLibraries ${ProjectLibraries} Qt5::Widgets)

# Include VTK.
find_package       (VTK REQUIRED)
include            (${VTK_USE_FILE})
include_directories(${VTK_INCLUDE_DIRS})
set(ProjectLibraries ${ProjectLibraries} ${VTK_LIBRARIES})


