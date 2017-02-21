#include /* implements */ <ui/viewer.hpp>

#include <cuda/sample.h>

namespace pli
{
viewer::viewer(QWidget* parent) : QOpenGLWidget(parent)
{
  makeCurrent();
  QOpenGLFunctions_4_5_Core::initializeOpenGLFunctions();
}

  void viewer::initializeGL()
  {
  }

  void viewer::paintGL()
  {
  }

  void viewer::resizeGL(int w, int h)
  {
  }
}
