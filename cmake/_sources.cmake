set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/aspects/loggable.hpp
  include/aspects/renderable.hpp
  include/cuda/sh/choose.h
  include/cuda/sh/clebsch_gordan.h
  include/cuda/sh/convert.h
  include/cuda/sh/cush.h
  include/cuda/sh/decorators.h
  include/cuda/sh/factorial.h
  include/cuda/sh/launch.h
  include/cuda/sh/legendre.h
  include/cuda/sh/sign.h
  include/cuda/sh/spherical_harmonics.h
  include/cuda/sh/vector_ops.h
  include/cuda/sh/wigner.h
  include/cuda/odf_field.h
  include/cuda/orthtree.h
  include/cuda/spherical_histogram.h
  include/cuda/vector_field.h
  include/io/hdf5_io.hpp
  include/io/hdf5_io_2.hpp
  include/io/hdf5_io_base.hpp
  include/io/hdf5_io_selector.hpp
  include/math/camera.hpp
  include/math/linear_math.hpp
  include/math/transform.hpp
  include/opengl/auxiliary/glm_uniforms.hpp
  include/opengl/all.hpp
  include/opengl/buffer.hpp
  include/opengl/framebuffer.hpp
  include/opengl/opengl.hpp
  include/opengl/program.hpp
  include/opengl/shader.hpp
  include/opengl/texture.hpp
  include/opengl/vertex_array.hpp
  include/third_party/glew/GL/glew.h
  include/third_party/qwt/qwt.h
  include/third_party/qwt/qwt_abstract_legend.h
  include/third_party/qwt/qwt_abstract_scale.h
  include/third_party/qwt/qwt_abstract_scale_draw.h
  include/third_party/qwt/qwt_abstract_slider.h
  include/third_party/qwt/qwt_analog_clock.h
  include/third_party/qwt/qwt_arrow_button.h
  include/third_party/qwt/qwt_clipper.h
  include/third_party/qwt/qwt_color_map.h
  include/third_party/qwt/qwt_column_symbol.h
  include/third_party/qwt/qwt_compass.h
  include/third_party/qwt/qwt_compass_rose.h
  include/third_party/qwt/qwt_compat.h
  include/third_party/qwt/qwt_counter.h
  include/third_party/qwt/qwt_curve_fitter.h
  include/third_party/qwt/qwt_date.h
  include/third_party/qwt/qwt_date_scale_draw.h
  include/third_party/qwt/qwt_date_scale_engine.h
  include/third_party/qwt/qwt_dial.h
  include/third_party/qwt/qwt_dial_needle.h
  include/third_party/qwt/qwt_dyngrid_layout.h
  include/third_party/qwt/qwt_event_pattern.h
  include/third_party/qwt/qwt_global.h
  include/third_party/qwt/qwt_graphic.h
  include/third_party/qwt/qwt_interval.h
  include/third_party/qwt/qwt_interval_symbol.h
  include/third_party/qwt/qwt_knob.h
  include/third_party/qwt/qwt_legend.h
  include/third_party/qwt/qwt_legend_data.h
  include/third_party/qwt/qwt_legend_label.h
  include/third_party/qwt/qwt_magnifier.h
  include/third_party/qwt/qwt_math.h
  include/third_party/qwt/qwt_matrix_raster_data.h
  include/third_party/qwt/qwt_null_paintdevice.h
  include/third_party/qwt/qwt_painter.h
  include/third_party/qwt/qwt_painter_command.h
  include/third_party/qwt/qwt_panner.h
  include/third_party/qwt/qwt_picker.h
  include/third_party/qwt/qwt_picker_machine.h
  include/third_party/qwt/qwt_pixel_matrix.h
  include/third_party/qwt/qwt_plot.h
  include/third_party/qwt/qwt_plot_abstract_barchart.h
  include/third_party/qwt/qwt_plot_barchart.h
  include/third_party/qwt/qwt_plot_canvas.h
  include/third_party/qwt/qwt_plot_curve.h
  include/third_party/qwt/qwt_plot_dict.h
  include/third_party/qwt/qwt_plot_directpainter.h
  include/third_party/qwt/qwt_plot_glcanvas.h
  include/third_party/qwt/qwt_plot_grid.h
  include/third_party/qwt/qwt_plot_histogram.h
  include/third_party/qwt/qwt_plot_intervalcurve.h
  include/third_party/qwt/qwt_plot_item.h
  include/third_party/qwt/qwt_plot_layout.h
  include/third_party/qwt/qwt_plot_legenditem.h
  include/third_party/qwt/qwt_plot_magnifier.h
  include/third_party/qwt/qwt_plot_marker.h
  include/third_party/qwt/qwt_plot_multi_barchart.h
  include/third_party/qwt/qwt_plot_panner.h
  include/third_party/qwt/qwt_plot_picker.h
  include/third_party/qwt/qwt_plot_rasteritem.h
  include/third_party/qwt/qwt_plot_renderer.h
  include/third_party/qwt/qwt_plot_rescaler.h
  include/third_party/qwt/qwt_plot_scaleitem.h
  include/third_party/qwt/qwt_plot_seriesitem.h
  include/third_party/qwt/qwt_plot_shapeitem.h
  include/third_party/qwt/qwt_plot_spectrocurve.h
  include/third_party/qwt/qwt_plot_spectrogram.h
  include/third_party/qwt/qwt_plot_svgitem.h
  include/third_party/qwt/qwt_plot_textlabel.h
  include/third_party/qwt/qwt_plot_tradingcurve.h
  include/third_party/qwt/qwt_plot_zoneitem.h
  include/third_party/qwt/qwt_plot_zoomer.h
  include/third_party/qwt/qwt_point_3d.h
  include/third_party/qwt/qwt_point_data.h
  include/third_party/qwt/qwt_point_mapper.h
  include/third_party/qwt/qwt_point_polar.h
  include/third_party/qwt/qwt_raster_data.h
  include/third_party/qwt/qwt_round_scale_draw.h
  include/third_party/qwt/qwt_samples.h
  include/third_party/qwt/qwt_sampling_thread.h
  include/third_party/qwt/qwt_scale_div.h
  include/third_party/qwt/qwt_scale_draw.h
  include/third_party/qwt/qwt_scale_engine.h
  include/third_party/qwt/qwt_scale_map.h
  include/third_party/qwt/qwt_scale_widget.h
  include/third_party/qwt/qwt_series_data.h
  include/third_party/qwt/qwt_series_store.h
  include/third_party/qwt/qwt_slider.h
  include/third_party/qwt/qwt_spline.h
  include/third_party/qwt/qwt_symbol.h
  include/third_party/qwt/qwt_system_clock.h
  include/third_party/qwt/qwt_text.h
  include/third_party/qwt/qwt_text_engine.h
  include/third_party/qwt/qwt_text_label.h
  include/third_party/qwt/qwt_thermo.h
  include/third_party/qwt/qwt_transform.h
  include/third_party/qwt/qwt_wheel.h
  include/third_party/qwt/qwt_widget_overlay.h
  include/third_party/qxt/QxtGlobal.h
  include/third_party/qxt/QxtLetterBoxWidget.h
  include/third_party/qxt/QxtLetterBoxWidgetP.h
  include/third_party/qxt/QxtSpanSlider.h
  include/third_party/qxt/QxtSpanSliderP.h
  include/third_party/tangent-base/analytic_orbit_interpolator.hpp
  include/third_party/tangent-base/base_operations.hpp
  include/third_party/tangent-base/base_types.hpp
  include/third_party/tangent-base/basic_trilinear_interpolator.hpp
  include/third_party/tangent-base/cartesian_grid.hpp
  include/third_party/tangent-base/cartesian_locator.hpp
  include/third_party/tangent-base/default_tracers.hpp
  include/third_party/tangent-base/dummy_recorder.hpp
  include/third_party/tangent-base/omp_pos_tracer.hpp
  include/third_party/tangent-base/particle_population.hpp
  include/third_party/tangent-base/raw_binary_reader.hpp
  include/third_party/tangent-base/runge_kutta_4_integrator.hpp
  include/third_party/tangent-base/simple_tracer.hpp
  include/third_party/tangent-base/trace_recorder.hpp
  include/third_party/tangent-base/tracer_base.hpp
  include/ui/plugins/data_plugin.hpp
  include/ui/plugins/fom_plugin.hpp
  include/ui/plugins/fdm_plugin.hpp
  include/ui/plugins/interactor_plugin.hpp
  include/ui/plugins/plugin.hpp
  include/ui/plugins/scalar_plugin.hpp
  include/ui/plugins/selector_plugin.hpp
  include/ui/plugins/tractography_plugin.hpp
  include/ui/plugins/volume_rendering_plugin.hpp
  include/ui/widgets/picker.hpp
  include/ui/widgets/transfer_function_widget.hpp
  include/ui/overview_image.hpp
  include/ui/selection_square.hpp
  include/ui/viewer.hpp
  include/ui/wait_spinner.hpp
  include/ui/window.hpp
  include/utility/line_edit_utility.hpp
  include/utility/qt_text_browser_sink.hpp
  include/utility/thread_pool.hpp
  include/visualization/interactors/first_person_interactor.hpp
  include/visualization/interactors/orbit_interactor.hpp
  include/visualization/interactors/simple_interactor.hpp
  include/visualization/basic_tracer.hpp
  include/visualization/odf_field.hpp
  include/visualization/scalar_field.hpp
  include/visualization/vector_field.hpp
  include/visualization/volume_renderer.hpp
  
  shaders/odf_field.frag.glsl
  shaders/odf_field.vert.glsl
  shaders/scalar_field.frag.glsl
  shaders/scalar_field.vert.glsl
  shaders/vector_field.frag.glsl
  shaders/vector_field.vert.glsl
  shaders/volume_renderer_prepass.frag.glsl
  shaders/volume_renderer_prepass.vert.glsl
  shaders/volume_renderer.frag.glsl
  shaders/volume_renderer.vert.glsl

  source/cuda/odf_field.cu
  source/cuda/vector_field.cu
  source/math/camera.cpp
  source/math/transform.cpp
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
  source/ui/plugins/fdm_plugin.cpp
  source/ui/plugins/interactor_plugin.cpp
  source/ui/plugins/plugin.cpp
  source/ui/plugins/scalar_plugin.cpp
  source/ui/plugins/selector_plugin.cpp
  source/ui/plugins/tractography_plugin.cpp
  source/ui/plugins/volume_rendering_plugin.cpp
  source/ui/widgets/picker.cpp
  source/ui/widgets/transfer_function_widget.cpp
  source/ui/overview_image.cpp
  source/ui/selection_square.cpp
  source/ui/viewer.cpp
  source/ui/wait_spinner.cpp
  source/ui/window.cpp
  source/visualization/interactors/first_person_interactor.cpp
  source/visualization/interactors/orbit_interactor.cpp
  source/visualization/interactors/simple_interactor.cpp
  source/visualization/basic_tracer.cpp
  source/visualization/odf_field.cpp
  source/visualization/scalar_field.cpp
  source/visualization/vector_field.cpp
  source/visualization/volume_renderer.cpp
  source/main.cpp
)
