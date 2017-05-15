#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <memory>
#include <vector>

#include <opengl/opengl.hpp>

#include <QOpenGLWidget>

#include <attributes/loggable.hpp>
#include <attributes/renderable.hpp>
#include <math/camera.hpp>
#include <ui/wait_spinner.hpp>
#include <visualization/interactors/simple_interactor.hpp>

namespace pli
{
class viewer : public QOpenGLWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);
  
  template<typename type, typename ...args>
  type* add_renderable   (args&&...   arguments );
  void  remove_renderable(renderable* renderable);

  camera*            camera    () { return &camera_    ; }
  simple_interactor* interactor() { return &interactor_; }

  void initializeGL   ()                   override;
  void paintGL        ()                   override;
  void resizeGL       (int w, int h)       override;
  void keyPressEvent  (QKeyEvent*   event) override;
  void keyReleaseEvent(QKeyEvent*   event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent (QMouseEvent* event) override;

  void set_wait_spinner_enabled(bool enabled) const;

private:
  bool                                     initialized_ = false;
  std::vector<std::unique_ptr<renderable>> renderables_ ;
  pli::camera                              camera_      ;
  pli::simple_interactor                   interactor_;
  pli::wait_spinner*                       wait_spinner_;
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
