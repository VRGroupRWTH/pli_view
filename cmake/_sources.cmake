set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/attributes/loggable.hpp
  include/attributes/renderable.hpp
  include/cuda/launch.h
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
  include/sh/choose.h
  include/sh/clebsch_gordan.h
  include/sh/convert.h
  include/sh/cush.h
  include/sh/decorators.h
  include/sh/factorial.h
  include/sh/launch.h
  include/sh/legendre.h
  include/sh/sign.h
  include/sh/spherical_harmonics.h
  include/sh/vector_ops.h
  include/sh/wigner.h
  include/third_party/qxt/qxtglobal.h
  include/third_party/qxt/qxtletterboxwidget.h
  include/third_party/qxt/qxtletterboxwidgetp.h
  include/third_party/qxt/qxtspanslider.h
  include/third_party/qxt/qxtspansliderp.h
  include/ui/plugins/data_plugin.hpp
  include/ui/plugins/fom_plugin.hpp
  include/ui/plugins/fdm_plugin.hpp
  include/ui/plugins/interactor_plugin.hpp
  include/ui/plugins/plugin.hpp
  include/ui/plugins/scalar_plugin.hpp
  include/ui/plugins/selector_plugin.hpp
  include/ui/plugins/tractography_plugin.hpp
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
  include/visualization/linear_tracer.hpp
  include/visualization/odf_field.hpp
  include/visualization/scalar_field.hpp
  include/visualization/vector_field.hpp
  
  shaders/odf_field.frag.glsl
  shaders/odf_field.vert.glsl
  shaders/scalar_field.frag.glsl
  shaders/scalar_field.vert.glsl
  shaders/vector_field.frag.glsl
  shaders/vector_field.vert.glsl

  source/cuda/odf_field.cu
  source/cuda/vector_field.cu
  source/math/camera.cpp
  source/math/transform.cpp
  source/third_party/qxt/qxtletterboxwidget.cpp
  source/third_party/qxt/qxtspanslider.cpp
  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/fdm_plugin.cpp
  source/ui/plugins/interactor_plugin.cpp
  source/ui/plugins/plugin.cpp
  source/ui/plugins/scalar_plugin.cpp
  source/ui/plugins/selector_plugin.cpp
  source/ui/plugins/tractography_plugin.cpp
  source/ui/overview_image.cpp
  source/ui/selection_square.cpp
  source/ui/viewer.cpp
  source/ui/wait_spinner.cpp
  source/ui/window.cpp
  source/visualization/interactors/first_person_interactor.cpp
  source/visualization/interactors/orbit_interactor.cpp
  source/visualization/interactors/simple_interactor.cpp
  source/visualization/odf_field.cpp
  source/visualization/scalar_field.cpp
  source/visualization/vector_field.cpp
  source/main.cpp
)
