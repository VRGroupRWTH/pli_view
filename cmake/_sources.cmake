set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/attributes/loggable.hpp
  
  include/cuda/convert.h
  include/cuda/decorators.h
  include/cuda/sampler.h

  include/graphics/color_mappers/rgb.hpp
  include/graphics/color_mappers/hsl.hpp
  include/graphics/color_mappers/hsv.hpp
  include/graphics/color_convert.hpp
  include/graphics/fdm_factory.hpp
  include/graphics/fom_factory.hpp
  include/graphics/sampling.hpp

  include/ui/plugins/data_plugin.hpp
  include/ui/plugins/fom_plugin.hpp
  include/ui/plugins/fdm_plugin.hpp
  include/ui/plugins/plugin.hpp
  include/ui/viewer.hpp
  include/ui/window.hpp
  
  include/utility/base_type.hpp
  include/utility/line_edit_utility.hpp
  include/utility/qt_text_browser_sink.hpp
  
  source/cuda/sampler.cu

  source/graphics/sampling.cpp

  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/fdm_plugin.cpp
  source/ui/plugins/plugin.cpp
  source/ui/viewer.cpp
  source/ui/window.cpp

  source/main.cpp
)
