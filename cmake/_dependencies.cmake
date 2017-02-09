# Register bundled modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

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



# Include cush.
find_package       (Cush REQUIRED)
include_directories(${CUSH_INCLUDE_DIRS})

# Include pli_io.
find_package       (PLI_IO REQUIRED)
include_directories(${PLI_IO_INCLUDE_DIRS})
