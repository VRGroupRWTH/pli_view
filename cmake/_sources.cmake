set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/attributes/loggable.hpp
  include/ui/plugins/data_plugin.hpp
  include/ui/plugins/fom_plugin.hpp
  include/ui/plugins/fdm_plugin.hpp
  include/ui/plugins/plugin.hpp
  include/ui/viewer.hpp
  include/ui/window.hpp
  include/utility/qt/line_edit_utility.hpp
  include/utility/spdlog/qt_text_browser_sink.hpp
  include/utility/std/base_type.hpp
  include/utility/vtk/color_mappers/rgb.hpp
  include/utility/vtk/fdm_factory.hpp
  include/utility/vtk/fom_factory.hpp
  
  source/ui/plugins/data_plugin.cpp
  source/ui/plugins/fom_plugin.cpp
  source/ui/plugins/fdm_plugin.cpp
  source/ui/plugins/plugin.cpp
  source/ui/viewer.cpp
  source/ui/window.cpp
  source/main.cpp
)
