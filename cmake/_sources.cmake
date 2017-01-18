set(ProjectSources
  cmake/_dependencies.cmake
  cmake/_sources.cmake
  
  include/adapters/qt/line_edit_utility.hpp
  include/adapters/spdlog/qt_text_browser_sink.hpp
  include/adapters/vtk/color_mappers/rgb.hpp
  include/adapters/vtk/fom_factory.hpp
  include/attributes/loggable.hpp
  include/ui/toolboxes/data_toolbox_widget.hpp
  include/ui/toolboxes/fom_toolbox_widget.hpp
  include/ui/toolboxes/fdm_toolbox_widget.hpp
  include/ui/window.hpp
  
  source/ui/toolboxes/data_toolbox_widget.cpp
  source/ui/toolboxes/fom_toolbox_widget.cpp
  source/ui/toolboxes/fdm_toolbox_widget.cpp
  source/ui/window.cpp
  source/main.cpp
)
