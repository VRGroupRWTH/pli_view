#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <all.hpp>

#include <QOpenGLWidget.h>
#include <QOpenGLFunctions_4_5_Core.h>

#include <attributes/loggable.hpp>

namespace pli
{
class viewer : public QOpenGLWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);

  void initializeGL()             override;
  void paintGL     ()             override;
  void resizeGL    (int w, int h) override;
};
}

#endif
