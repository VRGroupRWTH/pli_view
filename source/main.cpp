#include <window.hpp>

void main(int argc, char** argv)
{
  QApplication application(argc, argv);
  pli::window window;
  application.exec();
}