#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")

#include <QApplication.h>

#include <ui/window.hpp>

extern "C" 
{
  _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

void main(int argc, char** argv)
{
  QSurfaceFormat format;
  format.setProfile     (QSurfaceFormat::CompatibilityProfile);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer        );
  format.setSamples     (9);
  format.setVersion     (4, 5);
  QSurfaceFormat::setDefaultFormat(format);

  QApplication application(argc, argv);
  
  pli::window window;
  
  application.exec();
}