##################################################    Project     ##################################################
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
cmake_policy          (SET CMP0023 OLD)
project               (pli_vis VERSION 1.0 LANGUAGES C CXX)
list                  (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set_property          (GLOBAL PROPERTY USE_FOLDERS ON)
set                   (CMAKE_CXX_VISIBILITY_PRESET hidden)
set                   (CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set                   (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -fopenmp -lgomp")
set                   (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda"
                                                          "-lcublas"
                                                          "-lcublas_device"
                                                          "-lcudadevrt"
                                                         #"-gencode=arch=compute_50,code=sm_50")
                                                          "-gencode=arch=compute_60,code=sm_60")

##################################################    Options     ##################################################
option(BUILD_TESTS "Build tests." OFF)

##################################################    Sources     ##################################################
set(PROJECT_SOURCES
  CMakeLists.txt
  cmake/assign_source_group.cmake
  cmake/import_library.cmake
  
  include/pli_vis/aspects/loggable.hpp
  include/pli_vis/aspects/renderable.hpp
  include/pli_vis/cuda/mcmc/markov_chain.h
  include/pli_vis/cuda/mcmc/monte_carlo.h
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
  include/pli_vis/cuda/global_tractography.h
  include/pli_vis/cuda/odf_field.h
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
  include/pli_vis/third_party/qwt/qwt.h
  include/pli_vis/third_party/qwt/qwt_abstract_legend.h
  include/pli_vis/third_party/qwt/qwt_abstract_scale.h
  include/pli_vis/third_party/qwt/qwt_abstract_scale_draw.h
  include/pli_vis/third_party/qwt/qwt_abstract_slider.h
  include/pli_vis/third_party/qwt/qwt_analog_clock.h
  include/pli_vis/third_party/qwt/qwt_arrow_button.h
  include/pli_vis/third_party/qwt/qwt_clipper.h
  include/pli_vis/third_party/qwt/qwt_color_map.h
  include/pli_vis/third_party/qwt/qwt_column_symbol.h
  include/pli_vis/third_party/qwt/qwt_compass.h
  include/pli_vis/third_party/qwt/qwt_compass_rose.h
  include/pli_vis/third_party/qwt/qwt_compat.h
  include/pli_vis/third_party/qwt/qwt_counter.h
  include/pli_vis/third_party/qwt/qwt_curve_fitter.h
  include/pli_vis/third_party/qwt/qwt_date.h
  include/pli_vis/third_party/qwt/qwt_date_scale_draw.h
  include/pli_vis/third_party/qwt/qwt_date_scale_engine.h
  include/pli_vis/third_party/qwt/qwt_dial.h
  include/pli_vis/third_party/qwt/qwt_dial_needle.h
  include/pli_vis/third_party/qwt/qwt_dyngrid_layout.h
  include/pli_vis/third_party/qwt/qwt_event_pattern.h
  include/pli_vis/third_party/qwt/qwt_global.h
  include/pli_vis/third_party/qwt/qwt_graphic.h
  include/pli_vis/third_party/qwt/qwt_interval.h
  include/pli_vis/third_party/qwt/qwt_interval_symbol.h
  include/pli_vis/third_party/qwt/qwt_knob.h
  include/pli_vis/third_party/qwt/qwt_legend.h
  include/pli_vis/third_party/qwt/qwt_legend_data.h
  include/pli_vis/third_party/qwt/qwt_legend_label.h
  include/pli_vis/third_party/qwt/qwt_magnifier.h
  include/pli_vis/third_party/qwt/qwt_math.h
  include/pli_vis/third_party/qwt/qwt_matrix_raster_data.h
  include/pli_vis/third_party/qwt/qwt_null_paintdevice.h
  include/pli_vis/third_party/qwt/qwt_painter.h
  include/pli_vis/third_party/qwt/qwt_painter_command.h
  include/pli_vis/third_party/qwt/qwt_panner.h
  include/pli_vis/third_party/qwt/qwt_picker.h
  include/pli_vis/third_party/qwt/qwt_picker_machine.h
  include/pli_vis/third_party/qwt/qwt_pixel_matrix.h
  include/pli_vis/third_party/qwt/qwt_plot.h
  include/pli_vis/third_party/qwt/qwt_plot_abstract_barchart.h
  include/pli_vis/third_party/qwt/qwt_plot_barchart.h
  include/pli_vis/third_party/qwt/qwt_plot_canvas.h
  include/pli_vis/third_party/qwt/qwt_plot_curve.h
  include/pli_vis/third_party/qwt/qwt_plot_dict.h
  include/pli_vis/third_party/qwt/qwt_plot_directpainter.h
  include/pli_vis/third_party/qwt/qwt_plot_glcanvas.h
  include/pli_vis/third_party/qwt/qwt_plot_grid.h
  include/pli_vis/third_party/qwt/qwt_plot_histogram.h
  include/pli_vis/third_party/qwt/qwt_plot_intervalcurve.h
  include/pli_vis/third_party/qwt/qwt_plot_item.h
  include/pli_vis/third_party/qwt/qwt_plot_layout.h
  include/pli_vis/third_party/qwt/qwt_plot_legenditem.h
  include/pli_vis/third_party/qwt/qwt_plot_magnifier.h
  include/pli_vis/third_party/qwt/qwt_plot_marker.h
  include/pli_vis/third_party/qwt/qwt_plot_multi_barchart.h
  include/pli_vis/third_party/qwt/qwt_plot_panner.h
  include/pli_vis/third_party/qwt/qwt_plot_picker.h
  include/pli_vis/third_party/qwt/qwt_plot_rasteritem.h
  include/pli_vis/third_party/qwt/qwt_plot_renderer.h
  include/pli_vis/third_party/qwt/qwt_plot_rescaler.h
  include/pli_vis/third_party/qwt/qwt_plot_scaleitem.h
  include/pli_vis/third_party/qwt/qwt_plot_seriesitem.h
  include/pli_vis/third_party/qwt/qwt_plot_shapeitem.h
  include/pli_vis/third_party/qwt/qwt_plot_spectrocurve.h
  include/pli_vis/third_party/qwt/qwt_plot_spectrogram.h
  include/pli_vis/third_party/qwt/qwt_plot_svgitem.h
  include/pli_vis/third_party/qwt/qwt_plot_textlabel.h
  include/pli_vis/third_party/qwt/qwt_plot_tradingcurve.h
  include/pli_vis/third_party/qwt/qwt_plot_zoneitem.h
  include/pli_vis/third_party/qwt/qwt_plot_zoomer.h
  include/pli_vis/third_party/qwt/qwt_point_3d.h
  include/pli_vis/third_party/qwt/qwt_point_data.h
  include/pli_vis/third_party/qwt/qwt_point_mapper.h
  include/pli_vis/third_party/qwt/qwt_point_polar.h
  include/pli_vis/third_party/qwt/qwt_raster_data.h
  include/pli_vis/third_party/qwt/qwt_round_scale_draw.h
  include/pli_vis/third_party/qwt/qwt_samples.h
  include/pli_vis/third_party/qwt/qwt_sampling_thread.h
  include/pli_vis/third_party/qwt/qwt_scale_div.h
  include/pli_vis/third_party/qwt/qwt_scale_draw.h
  include/pli_vis/third_party/qwt/qwt_scale_engine.h
  include/pli_vis/third_party/qwt/qwt_scale_map.h
  include/pli_vis/third_party/qwt/qwt_scale_widget.h
  include/pli_vis/third_party/qwt/qwt_series_data.h
  include/pli_vis/third_party/qwt/qwt_series_store.h
  include/pli_vis/third_party/qwt/qwt_slider.h
  include/pli_vis/third_party/qwt/qwt_spline.h
  include/pli_vis/third_party/qwt/qwt_symbol.h
  include/pli_vis/third_party/qwt/qwt_system_clock.h
  include/pli_vis/third_party/qwt/qwt_text.h
  include/pli_vis/third_party/qwt/qwt_text_engine.h
  include/pli_vis/third_party/qwt/qwt_text_label.h
  include/pli_vis/third_party/qwt/qwt_thermo.h
  include/pli_vis/third_party/qwt/qwt_transform.h
  include/pli_vis/third_party/qwt/qwt_wheel.h
  include/pli_vis/third_party/qwt/qwt_widget_overlay.h
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
  include/pli_vis/ui/plugins/data_plugin.hpp
  include/pli_vis/ui/plugins/fom_plugin.hpp
  include/pli_vis/ui/plugins/interactor_plugin.hpp
  include/pli_vis/ui/plugins/odf_plugin.hpp
  include/pli_vis/ui/plugins/scalar_plugin.hpp
  include/pli_vis/ui/plugins/tractography_plugin.hpp
  include/pli_vis/ui/plugins/volume_rendering_plugin.hpp
  include/pli_vis/ui/utility/line_edit.hpp
  include/pli_vis/ui/utility/plot_interactor.hpp
  include/pli_vis/ui/utility/text_browser_sink.hpp
  include/pli_vis/ui/widgets/roi_rectangle.hpp
  include/pli_vis/ui/widgets/roi_selector.hpp
  include/pli_vis/ui/widgets/transfer_function_editor.hpp
  include/pli_vis/ui/widgets/viewer.hpp
  include/pli_vis/ui/widgets/wait_spinner.hpp
  include/pli_vis/ui/application.hpp
  include/pli_vis/ui/plugin.hpp
  include/pli_vis/ui/plugin_base.hpp
  include/pli_vis/visualization/algorithms/lineao_streamline_renderer.hpp
  include/pli_vis/visualization/algorithms/odf_field.hpp
  include/pli_vis/visualization/algorithms/scalar_field.hpp
  include/pli_vis/visualization/algorithms/streamline_renderer.hpp
  include/pli_vis/visualization/algorithms/vector_field.hpp
  include/pli_vis/visualization/algorithms/volume_renderer.hpp
  include/pli_vis/visualization/interactors/first_person_interactor.hpp
  include/pli_vis/visualization/interactors/interactor.hpp
  include/pli_vis/visualization/interactors/orbit_interactor.hpp
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
  shaders/lineao_color_pass.frag.glsl
  shaders/lineao_color_pass.vert.glsl
  shaders/lineao_main_pass.frag.glsl
  shaders/lineao_main_pass.vert.glsl
  shaders/lineao_normal_depth_pass.frag.glsl
  shaders/lineao_normal_depth_pass.vert.glsl
  shaders/lineao_zoom_pass.frag.glsl
  shaders/lineao_zoom_pass.vert.glsl
  shaders/simple_color.frag.glsl
  shaders/simple_color.vert.glsl
  shaders/simple_texture.frag.glsl
  shaders/simple_texture.vert.glsl
  shaders/streamline_renderer.frag.glsl
  shaders/streamline_renderer.vert.glsl
  shaders/view_dependent_vector_field.frag.glsl
  shaders/view_dependent_vector_field.geom.glsl
  shaders/view_dependent_vector_field.vert.glsl
  shaders/view_dependent.frag.glsl
  shaders/view_dependent.vert.glsl
  shaders/volume_renderer_prepass.frag.glsl
  shaders/volume_renderer_prepass.vert.glsl
  shaders/volume_renderer.frag.glsl
  shaders/volume_renderer.vert.glsl
  
  source/cuda/pt/tracer.cu
  source/cuda/odf_field.cu
  source/third_party/glew/glew.c
  source/third_party/qwt/qwt_abstract_legend.cpp
  source/third_party/qwt/qwt_abstract_scale.cpp
  source/third_party/qwt/qwt_abstract_scale_draw.cpp
  source/third_party/qwt/qwt_abstract_slider.cpp
  source/third_party/qwt/qwt_analog_clock.cpp
  source/third_party/qwt/qwt_arrow_button.cpp
  source/third_party/qwt/qwt_clipper.cpp
  source/third_party/qwt/qwt_color_map.cpp
  source/third_party/qwt/qwt_column_symbol.cpp
  source/third_party/qwt/qwt_compass.cpp
  source/third_party/qwt/qwt_compass_rose.cpp
  source/third_party/qwt/qwt_counter.cpp
  source/third_party/qwt/qwt_curve_fitter.cpp
  source/third_party/qwt/qwt_date.cpp
  source/third_party/qwt/qwt_date_scale_draw.cpp
  source/third_party/qwt/qwt_date_scale_engine.cpp
  source/third_party/qwt/qwt_dial.cpp
  source/third_party/qwt/qwt_dial_needle.cpp
  source/third_party/qwt/qwt_dyngrid_layout.cpp
  source/third_party/qwt/qwt_event_pattern.cpp
  source/third_party/qwt/qwt_graphic.cpp
  source/third_party/qwt/qwt_interval.cpp
  source/third_party/qwt/qwt_interval_symbol.cpp
  source/third_party/qwt/qwt_knob.cpp
  source/third_party/qwt/qwt_legend.cpp
  source/third_party/qwt/qwt_legend_data.cpp
  source/third_party/qwt/qwt_legend_label.cpp
  source/third_party/qwt/qwt_magnifier.cpp
  source/third_party/qwt/qwt_math.cpp
  source/third_party/qwt/qwt_matrix_raster_data.cpp
  source/third_party/qwt/qwt_null_paintdevice.cpp
  source/third_party/qwt/qwt_painter.cpp
  source/third_party/qwt/qwt_painter_command.cpp
  source/third_party/qwt/qwt_panner.cpp
  source/third_party/qwt/qwt_picker.cpp
  source/third_party/qwt/qwt_picker_machine.cpp
  source/third_party/qwt/qwt_pixel_matrix.cpp
  source/third_party/qwt/qwt_plot.cpp
  source/third_party/qwt/qwt_plot_abstract_barchart.cpp
  source/third_party/qwt/qwt_plot_axis.cpp
  source/third_party/qwt/qwt_plot_barchart.cpp
  source/third_party/qwt/qwt_plot_canvas.cpp
  source/third_party/qwt/qwt_plot_curve.cpp
  source/third_party/qwt/qwt_plot_dict.cpp
  source/third_party/qwt/qwt_plot_directpainter.cpp
  source/third_party/qwt/qwt_plot_glcanvas.cpp
  source/third_party/qwt/qwt_plot_grid.cpp
  source/third_party/qwt/qwt_plot_histogram.cpp
  source/third_party/qwt/qwt_plot_intervalcurve.cpp
  source/third_party/qwt/qwt_plot_item.cpp
  source/third_party/qwt/qwt_plot_layout.cpp
  source/third_party/qwt/qwt_plot_legenditem.cpp
  source/third_party/qwt/qwt_plot_magnifier.cpp
  source/third_party/qwt/qwt_plot_marker.cpp
  source/third_party/qwt/qwt_plot_multi_barchart.cpp
  source/third_party/qwt/qwt_plot_panner.cpp
  source/third_party/qwt/qwt_plot_picker.cpp
  source/third_party/qwt/qwt_plot_rasteritem.cpp
  source/third_party/qwt/qwt_plot_renderer.cpp
  source/third_party/qwt/qwt_plot_rescaler.cpp
  source/third_party/qwt/qwt_plot_scaleitem.cpp
  source/third_party/qwt/qwt_plot_seriesitem.cpp
  source/third_party/qwt/qwt_plot_shapeitem.cpp
  source/third_party/qwt/qwt_plot_spectrocurve.cpp
  source/third_party/qwt/qwt_plot_spectrogram.cpp
  source/third_party/qwt/qwt_plot_svgitem.cpp
  source/third_party/qwt/qwt_plot_textlabel.cpp
  source/third_party/qwt/qwt_plot_tradingcurve.cpp
  source/third_party/qwt/qwt_plot_xml.cpp
  source/third_party/qwt/qwt_plot_zoneitem.cpp
  source/third_party/qwt/qwt_plot_zoomer.cpp
  source/third_party/qwt/qwt_point_3d.cpp
  source/third_party/qwt/qwt_point_data.cpp
  source/third_party/qwt/qwt_point_mapper.cpp
  source/third_party/qwt/qwt_point_polar.cpp
  source/third_party/qwt/qwt_raster_data.cpp
  source/third_party/qwt/qwt_round_scale_draw.cpp
  source/third_party/qwt/qwt_sampling_thread.cpp
  source/third_party/qwt/qwt_scale_div.cpp
  source/third_party/qwt/qwt_scale_draw.cpp
  source/third_party/qwt/qwt_scale_engine.cpp
  source/third_party/qwt/qwt_scale_map.cpp
  source/third_party/qwt/qwt_scale_widget.cpp
  source/third_party/qwt/qwt_series_data.cpp
  source/third_party/qwt/qwt_slider.cpp
  source/third_party/qwt/qwt_spline.cpp
  source/third_party/qwt/qwt_symbol.cpp
  source/third_party/qwt/qwt_system_clock.cpp
  source/third_party/qwt/qwt_text.cpp
  source/third_party/qwt/qwt_text_engine.cpp
  source/third_party/qwt/qwt_text_label.cpp
  source/third_party/qwt/qwt_thermo.cpp
  source/third_party/qwt/qwt_transform.cpp
  source/third_party/qwt/qwt_wheel.cpp
  source/third_party/qwt/qwt_widget_overlay.cpp
  source/third_party/qxt/QxtLetterBoxWidget.cpp
  source/third_party/qxt/QxtSpanSlider.cpp
  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/interactor_plugin.cpp
  source/ui/plugins/odf_plugin.cpp
  source/ui/plugins/scalar_plugin.cpp
  source/ui/plugins/tractography_plugin.cpp
  source/ui/plugins/volume_rendering_plugin.cpp
  source/ui/utility/plot_interactor.cpp
  source/ui/widgets/roi_rectangle.cpp
  source/ui/widgets/roi_selector.cpp
  source/ui/widgets/transfer_function_editor.cpp
  source/ui/widgets/viewer.cpp
  source/ui/widgets/wait_spinner.cpp
  source/ui/application.cpp
  source/visualization/algorithms/lineao_streamline_renderer.cpp
  source/visualization/algorithms/odf_field.cpp
  source/visualization/algorithms/scalar_field.cpp
  source/visualization/algorithms/streamline_renderer.cpp
  source/visualization/algorithms/vector_field.cpp
  source/visualization/algorithms/volume_renderer.cpp
  source/visualization/interactors/first_person_interactor.cpp
  source/visualization/interactors/orbit_interactor.cpp
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

find_package  (OpenMP)
list          (APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})

find_package  (OpenGL REQUIRED)
import_library(opengl OPENGL_INCLUDE_DIRS OPENGL_LIBRARIES)

set           (CMAKE_AUTOMOC ON)
set           (CMAKE_AUTOUIC ON)
set           (CMAKE_INCLUDE_CURRENT_DIR ON)
find_package  (Qt5Widgets REQUIRED)
list          (APPEND PROJECT_LIBRARIES Qt5::Widgets)

##################################################    Targets     ##################################################
cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/third_party>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include> 
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}/third_party>
  PRIVATE source)
target_link_libraries     (${PROJECT_NAME} PUBLIC ${PROJECT_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DQWT_NO_SVG -DQT_NO_SVG -DQT_NO_PRINTER -DQT_NO_PRINTDIALOG)
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
