#include <pli_vis/visualization/interactors/orbit_interactor.hpp>

#include <QKeyEvent>

#include <pli_vis/visualization/primitives/camera.hpp>

namespace pli
{
orbit_interactor::orbit_interactor(camera* camera) : interactor(camera)
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
    auto translation = camera_->translation();
    camera_->translate(-translation);
    camera_->rotate   (glm::angleAxis(glm::radians( look_speed_ * dx), glm::vec3(0.0, 0.0, 1.0)));
    camera_->rotate   (glm::angleAxis(glm::radians(-look_speed_ * dy), camera_->right()));
    camera_->translate(glm::length(translation) * camera_->forward());
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
