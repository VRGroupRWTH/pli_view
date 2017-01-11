# Register bundled modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Include pli_io.
include_directories(${pli_io_INCLUDE_DIR})
set(ProjectLibraries ${ProjectLibraries} ${pli_io_LIBRARIES})
