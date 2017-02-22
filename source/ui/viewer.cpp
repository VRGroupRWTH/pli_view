#include /* implements */ <ui/viewer.hpp>

namespace pli
{
viewer::viewer(QWidget* parent) : QOpenGLWidget(parent)
{

}

void viewer::remove_renderable(renderable* renderable)
{
  renderables_.erase(std::remove_if(
    renderables_.begin(), 
    renderables_.end  (),
    [renderable](const std::unique_ptr<pli::renderable>& obj)
    {
      return obj.get() == renderable;
    }), 
    renderables_.end  ());
}

void viewer::initializeGL()
{
  makeCurrent ();
  opengl::init();
  initialized_ = true;

  for (auto& renderable : renderables_)
    renderable->initialize();
}
void viewer::paintGL     ()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  for (auto& renderable : renderables_)
    renderable->render();
}
void viewer::resizeGL    (int w, int h)
{
  glViewport(0, 0, w, h);

  // TODO: Adjust projection matrix.
}
}
