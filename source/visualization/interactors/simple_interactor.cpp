#include <pli_vis/visualization/interactors/simple_interactor.hpp>

#include <QKeyEvent>

#include <pli_vis/visualization/primitives/camera.hpp>

namespace pli
{
simple_interactor::simple_interactor(camera* camera) : interactor(camera)
{
  
}

void simple_interactor::mouse_press_handler(QMouseEvent* event)
{
  last_mouse_position_ = event->pos();
}
void simple_interactor::mouse_move_handler (QMouseEvent* event)
{
  auto dx = event->x() - last_mouse_position_.x();
  auto dy = event->y() - last_mouse_position_.y();

  if (event->buttons() & Qt::LeftButton)
  {
    camera_->rotate(glm::angleAxis(glm::radians( look_speed_ * dx), glm::vec3(0.0, 0.0, 1.0)));
    camera_->rotate(glm::angleAxis(glm::radians(-look_speed_ * dy), camera_->right()));
  }
  if (event->buttons() & Qt::MiddleButton)
  {
    camera_->translate(move_speed_ * (float(dx) * camera_->right() - float(dy) * camera_->up()));
  }
  if (event->buttons() & Qt::RightButton)
  {
    if(camera_->orthographic())
      camera_->set_scale(camera_->scale() + glm::vec3(1.0) * ((1.0F / (2.0F * camera_->orthographic_size())) * move_speed_ * dy));
    else
      camera_->translate(move_speed_ * dy * camera_->forward());
  }

  last_mouse_position_ = event->pos();
}
}
