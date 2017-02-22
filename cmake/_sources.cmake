set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/attributes/loggable.hpp
  include/attributes/renderable.hpp
  include/cuda/sample.h
  include/cuda/vector_field.h
  include/graphics/color_mappers/rgb.hpp
  include/graphics/color_mappers/hsl.hpp
  include/graphics/color_mappers/hsv.hpp
  include/graphics/color_convert.hpp
  #include/graphics/fom_factory.hpp
  #include/graphics/fdm_factory.hpp
  include/graphics/vector_field.hpp
  include/ui/plugins/data_plugin.hpp
  include/ui/plugins/fom_plugin.hpp
  include/ui/plugins/fdm_plugin.hpp
  include/ui/plugins/plugin.hpp
  include/ui/viewer.hpp
  include/ui/window.hpp
  include/utility/line_edit_utility.hpp
  include/utility/qt_text_browser_sink.hpp
  
  shaders/vector_field.frag.glsl
  shaders/vector_field.vert.glsl

  source/cuda/sample.cu
  source/cuda/vector_field.cu
  source/graphics/vector_field.cpp
  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/fdm_plugin.cpp
  source/ui/plugins/plugin.cpp
  source/ui/viewer.cpp
  source/ui/window.cpp
  source/main.cpp
)
