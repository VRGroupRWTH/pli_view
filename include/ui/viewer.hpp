#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <memory>
#include <vector>

#include <opengl.hpp>
#include <QOpenGLWidget.h>

#include <attributes/loggable.hpp>
#include <attributes/renderable.hpp>

namespace pli
{
class viewer : public QOpenGLWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);

  template<typename type, typename ...args>
  type* add_renderable   (args&&...   arguments );
  void  remove_renderable(renderable* renderable);

  void initializeGL()             override;
  void paintGL     ()             override;
  void resizeGL    (int w, int h) override;

private:
  bool                                     initialized_ = false;
  std::vector<std::unique_ptr<renderable>> renderables_ ;
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
