#include <ui/window.hpp>

extern "C" 
{
  _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

void main(int argc, char** argv)
{
  QApplication application(argc, argv);
  pli::window  window;
  application.exec();
}