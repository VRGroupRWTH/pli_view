#include <pli_vis/ui/widgets/viewer.hpp>

#include <QKeyEvent>
#include <QTimer>

#include <pli_vis/visualization/interactors/simple_interactor.hpp>

namespace pli
{
viewer::viewer(QWidget* parent) : QOpenGLWidget(parent), interactor_(std::make_unique<simple_interactor>(&camera_)), wait_spinner_(new wait_spinner(this, true, false))
{
  reset_camera_transform();

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

void viewer::reset_camera_transform()
{
  camera_.set_orthographic_size(100);
  camera_.set_translation      ({ 0, 0, -100 });
  camera_.look_at              ({ 0, 0,    0 }, { 0, -1, 0 });
}

void viewer::initializeGL   ()
{
  makeCurrent ();
  opengl::init();
  initialized_ = true;

  for (auto& renderable : renderables_)
    renderable->initialize();

  // Make adjustible.
  glLineWidth(2);

  glEnable   (GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
}
void viewer::paintGL        ()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  interactor_->update_transform();
  for (auto& renderable : renderables_)
    if (renderable->active())
      renderable->render(&camera_);
}
void viewer::resizeGL       (int w, int h)
{
  glViewport(0, 0, w, h);
  camera_.set_aspect_ratio((float) w / h);
}
void viewer::keyPressEvent  (QKeyEvent*   event)
{
  interactor_->key_press_handler(event);
  update();
}
void viewer::keyReleaseEvent(QKeyEvent*   event)
{
  interactor_->key_release_handler(event);
  update();
}
void viewer::mousePressEvent(QMouseEvent* event)
{
  interactor_->mouse_press_handler(event);
  update();
}
void viewer::mouseMoveEvent (QMouseEvent* event)
{
  interactor_->mouse_move_handler(event);
}

void viewer::set_wait_spinner_enabled(bool enabled) const
{
  enabled ? wait_spinner_->start() : wait_spinner_->stop();
}
}
