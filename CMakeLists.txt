##################################################    Project     ##################################################
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
cmake_policy          (SET CMP0023 OLD)
project               (pli_vis VERSION 1.0 LANGUAGES C CXX)
list                  (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set_property          (GLOBAL PROPERTY USE_FOLDERS ON)
set                   (CMAKE_CXX_VISIBILITY_PRESET hidden)
set                   (CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set                   (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
set                   (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda"
                                                          "-lcublas"
                                                          "-lcudadevrt"
                                                          "-gencode=arch=compute_52,code=sm_52"
                                                          "-gencode=arch=compute_60,code=sm_60"
                                                          "-gencode=arch=compute_61,code=sm_61")
set                   (CUDA_SEPARABLE_COMPILATION ON)
set                   (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

##################################################    Options     ##################################################
option(BUILD_TESTS "Build tests." OFF)

##################################################    Sources     ##################################################
set(PROJECT_SOURCES
  CMakeLists.txt
  cmake/assign_source_group.cmake
  cmake/import_library.cmake
  
  include/pli_vis/aspects/loggable.hpp
  include/pli_vis/aspects/renderable.hpp
  include/pli_vis/cuda/pt/cartesian_locator.h
  include/pli_vis/cuda/pt/runge_kutta_4_integrator.h
  include/pli_vis/cuda/pt/tracer.h
  include/pli_vis/cuda/pt/trilinear_interpolator.h
  include/pli_vis/cuda/sh/choose.h
  include/pli_vis/cuda/sh/clebsch_gordan.h
  include/pli_vis/cuda/sh/factorial.h
  include/pli_vis/cuda/sh/launch.h
  include/pli_vis/cuda/sh/legendre.h
  include/pli_vis/cuda/sh/spherical_harmonics.h
  include/pli_vis/cuda/sh/wigner.h
  include/pli_vis/cuda/utility/convert.h
  include/pli_vis/cuda/utility/vector_ops.h
  include/pli_vis/cuda/odf_field.h
  include/pli_vis/cuda/polar_plot.h
  include/pli_vis/cuda/spherical_histogram.h
  include/pli_vis/io/io.hpp
  include/pli_vis/io/io_slice_impl.hpp
  include/pli_vis/io/io_volume_impl.hpp
  include/pli_vis/opengl/auxiliary/glm_uniforms.hpp
  include/pli_vis/opengl/all.hpp
  include/pli_vis/opengl/buffer.hpp
  include/pli_vis/opengl/framebuffer.hpp
  include/pli_vis/opengl/opengl.hpp
  include/pli_vis/opengl/program.hpp
  include/pli_vis/opengl/shader.hpp
  include/pli_vis/opengl/texture.hpp
  include/pli_vis/opengl/vertex_array.hpp
  include/pli_vis/third_party/glew/GL/glew.h
  include/pli_vis/third_party/json/json.hpp
  include/pli_vis/third_party/qxt/QxtGlobal.h
  include/pli_vis/third_party/qxt/QxtLetterBoxWidget.h
  include/pli_vis/third_party/qxt/QxtLetterBoxWidgetP.h
  include/pli_vis/third_party/qxt/QxtSpanSlider.h
  include/pli_vis/third_party/qxt/QxtSpanSliderP.h
  include/pli_vis/third_party/tangent-base/analytic_orbit_interpolator.hpp
  include/pli_vis/third_party/tangent-base/base_operations.hpp
  include/pli_vis/third_party/tangent-base/base_types.hpp
  include/pli_vis/third_party/tangent-base/basic_trilinear_interpolator.hpp
  include/pli_vis/third_party/tangent-base/cartesian_grid.hpp
  include/pli_vis/third_party/tangent-base/cartesian_locator.hpp
  include/pli_vis/third_party/tangent-base/default_tracers.hpp
  include/pli_vis/third_party/tangent-base/dummy_recorder.hpp
  include/pli_vis/third_party/tangent-base/omp_pos_tracer.hpp
  include/pli_vis/third_party/tangent-base/particle_population.hpp
  include/pli_vis/third_party/tangent-base/raw_binary_reader.hpp
  include/pli_vis/third_party/tangent-base/runge_kutta_4_integrator.hpp
  include/pli_vis/third_party/tangent-base/simple_tracer.hpp
  include/pli_vis/third_party/tangent-base/trace_recorder.hpp
  include/pli_vis/third_party/tangent-base/tracer_base.hpp
	include/pli_vis/ui/plugins/color_plugin.hpp
  include/pli_vis/ui/plugins/data_plugin.hpp
  include/pli_vis/ui/plugins/demo_plugin.hpp
  include/pli_vis/ui/plugins/fom_plugin.hpp
  include/pli_vis/ui/plugins/interactor_plugin.hpp
  include/pli_vis/ui/plugins/odf_plugin.hpp
  include/pli_vis/ui/plugins/polar_plot_plugin.hpp
  include/pli_vis/ui/plugins/scalar_plugin.hpp
  include/pli_vis/ui/plugins/local_tractography_plugin.hpp
  include/pli_vis/ui/utility/line_edit.hpp
  include/pli_vis/ui/utility/text_browser_sink.hpp
  include/pli_vis/ui/widgets/remote_viewer.hpp
  include/pli_vis/ui/widgets/roi_rectangle.hpp
  include/pli_vis/ui/widgets/roi_selector.hpp
  include/pli_vis/ui/widgets/viewer.hpp
  include/pli_vis/ui/widgets/wait_spinner.hpp
  include/pli_vis/ui/application.hpp
  include/pli_vis/ui/plugin.hpp
  include/pli_vis/ui/plugin_base.hpp
	include/pli_vis/utility/make_even.hpp
  include/pli_vis/visualization/algorithms/color_mapper.hpp
  include/pli_vis/visualization/algorithms/lineao_streamline_renderer.hpp
  include/pli_vis/visualization/algorithms/odf_field.hpp
  include/pli_vis/visualization/algorithms/polar_plot_field.hpp
  include/pli_vis/visualization/algorithms/scalar_field.hpp
  include/pli_vis/visualization/algorithms/streamline_renderer.hpp
  include/pli_vis/visualization/algorithms/vector_field.hpp
  include/pli_vis/visualization/interactors/first_person_interactor.hpp
  include/pli_vis/visualization/interactors/interactor.hpp
  include/pli_vis/visualization/interactors/simple_interactor.hpp
  include/pli_vis/visualization/primitives/camera.hpp
  include/pli_vis/visualization/primitives/directional_light.hpp
  include/pli_vis/visualization/primitives/light.hpp
  include/pli_vis/visualization/primitives/point_light.hpp
  include/pli_vis/visualization/primitives/spot_light.hpp
  include/pli_vis/visualization/primitives/transform.hpp
  include/pli_vis/visualization/utility/render_target.hpp
  include/pli_vis/visualization/utility/sphere_tessellation.hpp
  
  shaders/depth_pass.frag.glsl
  shaders/depth_pass.vert.glsl
  shaders/fullscreen_texture.frag.glsl
  shaders/fullscreen_texture.vert.glsl
  shaders/lineao_color_pass.frag.glsl
  shaders/lineao_color_pass.vert.glsl
  shaders/lineao_main_pass.frag.glsl
  shaders/lineao_main_pass.vert.glsl
  shaders/lineao_normal_depth_pass.frag.glsl
  shaders/lineao_normal_depth_pass.vert.glsl
  shaders/lineao_zoom_pass.frag.glsl
  shaders/lineao_zoom_pass.vert.glsl
  shaders/odf_field_renderer.frag.glsl
  shaders/odf_field_renderer.vert.glsl
  shaders/polar_plot.frag.glsl
  shaders/polar_plot.vert.glsl
  shaders/simple_color_texture.frag.glsl
  shaders/simple_color_texture.vert.glsl
  shaders/simple_color.frag.glsl
  shaders/simple_color.vert.glsl
  shaders/simple_texture.frag.glsl
  shaders/simple_texture.vert.glsl
  shaders/streamline_renderer.frag.glsl
  shaders/streamline_renderer.vert.glsl
  shaders/view_dependent.frag.glsl
  shaders/view_dependent.vert.glsl
  shaders/view_dependent_vector_field.frag.glsl
  shaders/view_dependent_vector_field.geom.glsl
  shaders/view_dependent_vector_field.vert.glsl
  
  source/cuda/pt/tracer.cu
  source/cuda/odf_field.cu
  source/cuda/polar_plot.cu
  source/third_party/glew/glew.c
  source/third_party/qxt/QxtLetterBoxWidget.cpp
  source/third_party/qxt/QxtSpanSlider.cpp
	source/ui/plugins/color_plugin.cpp
  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/demo_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/interactor_plugin.cpp
  source/ui/plugins/odf_plugin.cpp
  source/ui/plugins/polar_plot_plugin.cpp
  source/ui/plugins/scalar_plugin.cpp
  source/ui/plugins/local_tractography_plugin.cpp
  source/ui/widgets/remote_viewer.cpp
  source/ui/widgets/roi_rectangle.cpp
  source/ui/widgets/roi_selector.cpp
  source/ui/widgets/viewer.cpp
  source/ui/widgets/wait_spinner.cpp
  source/ui/application.cpp
  source/visualization/algorithms/lineao_streamline_renderer.cpp
  source/visualization/algorithms/odf_field.cpp
  source/visualization/algorithms/polar_plot_field.cpp
  source/visualization/algorithms/scalar_field.cpp
  source/visualization/algorithms/streamline_renderer.cpp
  source/visualization/algorithms/vector_field.cpp
  source/visualization/interactors/first_person_interactor.cpp
  source/visualization/interactors/simple_interactor.cpp
  source/visualization/primitives/camera.cpp
  source/visualization/primitives/directional_light.cpp
  source/visualization/primitives/light.cpp
  source/visualization/primitives/point_light.cpp
  source/visualization/primitives/spot_light.cpp
  source/visualization/primitives/transform.cpp
  source/visualization/utility/render_target.cpp
  source/main.cpp
)
include(assign_source_group)
assign_source_group(${PROJECT_SOURCES})

##################################################  Dependencies  ##################################################
include(import_library)

find_package  (Boost REQUIRED)
import_library(boost Boost_INCLUDE_DIR)

set           (CUDA_SEPARABLE_COMPILATION ON)
find_package  (CUDA REQUIRED)
list          (APPEND PROJECT_LIBRARIES 
               ${CUDA_CUBLAS_LIBRARIES} 
               ${CUDA_cusolver_LIBRARY} 
               ${CUDA_cudadevrt_LIBRARY} 
               ${CUDA_curand_LIBRARY})

find_package(HDF5 NAMES hdf5 COMPONENTS C shared)
if(NOT HDF5_INCLUDE_DIR OR NOT HDF5_C_SHARED_LIBRARY)
  set(HDF5_INCLUDE_DIR      CACHE STRING "")
  set(HDF5_C_SHARED_LIBRARY CACHE STRING "")
  message(FATAL_ERROR "HDF5 not found! Please set the include directory and libraries manually.")
else()
  import_library(hdf5 HDF5_INCLUDE_DIR HDF5_C_SHARED_LIBRARY)
endif()

find_package  (OpenGL REQUIRED)
import_library(opengl OPENGL_INCLUDE_DIRS OPENGL_LIBRARIES)

find_package  (Protobuf REQUIRED)
list          (APPEND PROJECT_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
list          (APPEND PROJECT_LIBRARIES    ${PROTOBUF_LIBRARIES}   )

find_package  (TBB REQUIRED)
import_library(tbb TBB_INCLUDE_DIRS TBB_LIBRARY_DEBUG TBB_LIBRARY_RELEASE)
import_library(tbb_malloc TBB_MALLOC_INCLUDE_DIRS TBB_MALLOC_LIBRARY_DEBUG TBB_MALLOC_LIBRARY_RELEASE)
import_library(tbb_malloc_proxy TBB_MALLOC_PROXY_INCLUDE_DIRS TBB_MALLOC_PROXY_LIBRARY_DEBUG TBB_MALLOC_PROXY_LIBRARY_RELEASE)

find_package  (ZeroMQ REQUIRED)
list          (APPEND PROJECT_INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR})
list          (APPEND PROJECT_LIBRARIES    ${ZeroMQ_LIBRARIES}  )

set           (CMAKE_AUTOMOC ON)
set           (CMAKE_AUTOUIC ON)
set           (CMAKE_INCLUDE_CURRENT_DIR ON)
find_package  (Qt5Widgets REQUIRED)
list          (APPEND PROJECT_LIBRARIES Qt5::Widgets)

##################################################    Prebuild    ##################################################
set(PROJECT_PROTO_SOURCES
  proto/camera.proto
  proto/color_mapping.proto
  proto/dataset_format.proto
  proto/image.proto
  proto/parameters.proto
  proto/selection.proto
  proto/vector_types.proto
)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROJECT_PROTO_SOURCES})
list                 (APPEND PROJECT_SOURCES ${PROJECT_PROTO_SOURCES} ${PROTO_HDRS} ${PROTO_SRCS})
assign_source_group  (${PROJECT_SOURCES})

##################################################    Targets     ##################################################
cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/third_party>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include> 
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}/third_party>
  PRIVATE source)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_INCLUDE_DIRS})
target_link_libraries     (${PROJECT_NAME} PUBLIC ${PROJECT_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DH5_BUILT_AS_DYNAMIC_LIB -DQWT_NO_SVG -DQT_NO_SVG -DQT_NO_PRINTER -DQT_NO_PRINTDIALOG)
set_target_properties     (${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

##################################################    Testing     ##################################################
if(BUILD_TESTS)
  enable_testing()

  set(PROJECT_TEST_SOURCES

  )

  foreach(_SOURCE ${PROJECT_TEST_SOURCES})
    get_filename_component(_NAME ${_SOURCE} NAME_WE)
    set                   (_SOURCES tests/catch.hpp tests/main.cpp ${_SOURCE})
    add_executable        (${_NAME} ${_SOURCES})
    target_link_libraries (${_NAME} ${PROJECT_NAME})
    add_test              (${_NAME} ${_NAME})
    set_property          (TARGET ${_NAME} PROPERTY FOLDER "Tests")
    source_group          ("source" FILES ${_SOURCES})
  endforeach()
endif()

##################################################  Installation  ##################################################
install(TARGETS ${PROJECT_NAME} EXPORT "${PROJECT_NAME}-config"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT  "${PROJECT_NAME}-config" DESTINATION "cmake")
export (TARGETS "${PROJECT_NAME}"        FILE        "${PROJECT_NAME}-config.cmake")
