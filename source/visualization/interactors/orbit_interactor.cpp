#include /* implements */ <visualization/interactors/orbit_interactor.hpp>

#include <QKeyEvent>

#include <math/transform.hpp>

namespace pli
{
orbit_interactor::orbit_interactor(transform* transform) : transform_(transform)
{
  
}

void orbit_interactor::update_transform   ()
{

}

void orbit_interactor::key_press_handler  (QKeyEvent*   event)
{

}
void orbit_interactor::key_release_handler(QKeyEvent*   event)
{

}
void orbit_interactor::mouse_press_handler(QMouseEvent* event)
{
  last_mouse_position_ = event->pos();
}
void orbit_interactor::mouse_move_handler (QMouseEvent* event)
{
  auto dx = event->x() - last_mouse_position_.x();
  auto dy = event->y() - last_mouse_position_.y();

  if (event->buttons() & Qt::LeftButton)
  {
    auto translation = transform_->translation();
    transform_->translate(-translation);
    transform_->rotate   (angleAxis(radians(-look_speed_ * dx), vec3f(0.0, 0.0, 1.0)));
    transform_->rotate   (angleAxis(radians(-look_speed_ * dy), transform_->right()));
    transform_->translate(length(translation) * transform_->forward());
  }
  if (event->buttons() & Qt::RightButton)
  {
    transform_->translate(move_speed_ * dy * transform_->forward());
  }

  last_mouse_position_ = event->pos();
}
}
