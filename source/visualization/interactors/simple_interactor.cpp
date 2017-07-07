#include <pli_vis/visualization/interactors/simple_interactor.hpp>

#include <QKeyEvent>

#include <pli_vis/visualization/transform.hpp>

namespace pli
{
simple_interactor::simple_interactor(transform* transform) : transform_(transform)
{
  
}

void simple_interactor::update_transform   ()
{

}

void simple_interactor::key_press_handler  (QKeyEvent*   event)
{

}
void simple_interactor::key_release_handler(QKeyEvent*   event)
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
    transform_->rotate(glm::angleAxis(glm::radians(-look_speed_ * dx), glm::vec3(0.0, 0.0, 1.0)));
    transform_->rotate(glm::angleAxis(glm::radians(-look_speed_ * dy), transform_->right()));
  }
  if (event->buttons() & Qt::MiddleButton)
  {
    transform_->translate(move_speed_ * (float(dx) * transform_->right() - float(dy) * transform_->up()));
  }
  if (event->buttons() & Qt::RightButton)
  {
    transform_->translate(move_speed_ * dy * transform_->forward());
  }

  last_mouse_position_ = event->pos();
}
}
