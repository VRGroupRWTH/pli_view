#ifdef _WIN32
#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
extern "C"
{
  _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

#include <QApplication>
#include <QSurfaceFormat>

#include <pli_vis/ui/application.hpp>

int main(int argc, char** argv)
{
  QSurfaceFormat format;
  format.setProfile     (QSurfaceFormat::CompatibilityProfile);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer        );
  format.setSamples     (9);
  format.setVersion     (4, 5);
  QSurfaceFormat::setDefaultFormat(format);

  QApplication application(argc, argv);
  
  pli::application window;
  
  application.exec();

  return 0;
}