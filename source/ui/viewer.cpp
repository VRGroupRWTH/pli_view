#include /* implements */ <ui/viewer.hpp>

#include <QKeyEvent>
#include <QTimer>

#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
viewer::viewer(QWidget* parent) : QOpenGLWidget(parent), interactor_(&camera_)
{
  // Make adjustable.
  interactor_.set_move_speed(0.001);

  setFocusPolicy(Qt::StrongFocus);

  auto timer = new QTimer(this);
  connect(timer, SIGNAL(timeout()), this, SLOT(update()));
  timer->start(16);
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

void viewer::initializeGL   ()
{
  makeCurrent ();
  opengl::init();
  initialized_ = true;

  for (auto& renderable : renderables_)
    renderable->initialize();

  // Make adjustible.
  glLineWidth(4);

  glEnable   (GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
}
void viewer::paintGL        ()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  interactor_.update_transform();
  for (auto& renderable : renderables_)
    renderable->render(&camera_);
}
void viewer::resizeGL       (int w, int h)
{
  glViewport(0, 0, w, h);
  camera_.set_aspect_ratio((float) w / h);
}
void viewer::keyPressEvent  (QKeyEvent* event)
{
  interactor_.key_press_handler(event);
  update();
}
void viewer::keyReleaseEvent(QKeyEvent* event)
{
  interactor_.key_release_handler(event);
  update();
}

}
