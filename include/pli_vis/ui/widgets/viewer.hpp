#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <memory>
#include <vector>

#include <pli_vis/opengl/opengl.hpp>

#include <QOpenGLWidget>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/visualization/interactors/interactor.hpp>
#include <pli_vis/visualization/primitives/camera.hpp>

namespace pli
{
class viewer : public QOpenGLWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);
  
  template<typename type, typename ...args>
  type* add_renderable   (args&&...   arguments );
  void  remove_renderable(renderable* renderable);

  template<typename interactor_type>
  void set_interactor()
  {
    interactor_ = std::make_unique<interactor_type>(&camera_);
  }

  camera*     camera    () { return &camera_         ; }
  interactor* interactor() { return interactor_.get(); }

  void reset_camera_transform();

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
  std::unique_ptr<pli::interactor>         interactor_  ;
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
