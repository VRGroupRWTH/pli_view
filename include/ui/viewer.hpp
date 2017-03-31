#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <map>
#include <memory>
#include <vector>

#include <opengl.hpp>
#include <QOpenGLWidget.h>

#include <attributes/loggable.hpp>
#include <attributes/renderable.hpp>
#include <math/camera.hpp>
#include <visualization/interactors/interactor.hpp>

namespace pli
{
class viewer : public QOpenGLWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);
  
  template<typename type, typename ...args>
  type* add_renderable   (args&&...   arguments );
  void  remove_renderable(renderable* renderable);

  camera*     camera    () { return &camera_    ; }
  interactor* interactor() { return &interactor_; }

  void initializeGL   ()                   override;
  void paintGL        ()                   override;
  void resizeGL       (int w, int h)       override;
  void keyPressEvent  (QKeyEvent*   event) override;
  void keyReleaseEvent(QKeyEvent*   event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent (QMouseEvent* event) override;

private:
  bool                                     initialized_ = false;
  std::vector<std::unique_ptr<renderable>> renderables_ ;
  pli::camera                              camera_      ;
  pli::interactor                          interactor_  ;
};

template <typename type, typename ... args>
type* viewer::add_renderable(args&&... arguments)
{
  renderables_.emplace_back(new type(arguments...));
  auto renderable = (type*) renderables_.back().get();

  if (initialized_)
    renderable->initialize();

  return renderable;
}
}

#endif
