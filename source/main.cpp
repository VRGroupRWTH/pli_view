#ifdef _WIN32
#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
extern "C"
{
  _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

#include <omp.h>
#include <QApplication>
#include <QSurfaceFormat>

#include <pli_vis/ui/window.hpp>

int main(int argc, char** argv)
{
  omp_set_num_threads(4);

  QSurfaceFormat format;
  format.setProfile     (QSurfaceFormat::CompatibilityProfile);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer        );
  format.setSamples     (9);
  format.setVersion     (4, 5);
  QSurfaceFormat::setDefaultFormat(format);

  QApplication application(argc, argv);
  
  pli::window window;
  
  application.exec();

  return 0;
}