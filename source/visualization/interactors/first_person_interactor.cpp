#include <pli_vis/visualization/interactors/first_person_interactor.hpp>

#include <QKeyEvent>

#include <pli_vis/visualization/transform.hpp>

namespace pli
{
first_person_interactor::first_person_interactor(transform* transform) : transform_(transform)
{

}

void first_person_interactor::update_transform()
{
  if (key_map_[Qt::Key_W])
    transform_->translate(- transform_->forward() * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
  if (key_map_[Qt::Key_A])
    transform_->translate(- transform_->right  () * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
  if (key_map_[Qt::Key_S])
    transform_->translate(  transform_->forward() * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
  if (key_map_[Qt::Key_D])
    transform_->translate(  transform_->right  () * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
  if (key_map_[Qt::Key_Z])
    transform_->translate(- transform_->up     () * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
  if (key_map_[Qt::Key_X])
    transform_->translate(  transform_->up     () * (move_speed_ * (key_map_[Qt::Key_Shift] ? 2 : 1)));
}

void first_person_interactor::key_press_handler  (QKeyEvent*   event)
{
  switch (event->key())
  {
  case Qt::Key_W:
    key_map_[Qt::Key_W] = true;
    break;
  case Qt::Key_A:
    key_map_[Qt::Key_A] = true;
    break;
  case Qt::Key_S:
    key_map_[Qt::Key_S] = true;
    break;
  case Qt::Key_D:
    key_map_[Qt::Key_D] = true;
    break;
  case Qt::Key_Z:
    key_map_[Qt::Key_Z] = true;
    break;
  case Qt::Key_X:
    key_map_[Qt::Key_X] = true;
    break;
  case Qt::Key_Shift:
    key_map_[Qt::Key_Shift] = true;
    break;
  default: 
    break;
  }
}
void first_person_interactor::key_release_handler(QKeyEvent*   event)
{
  switch (event->key())
  {
  case Qt::Key_W:
    key_map_[Qt::Key_W] = false;
    break;
  case Qt::Key_A:
    key_map_[Qt::Key_A] = false;
    break;
  case Qt::Key_S:
    key_map_[Qt::Key_S] = false;
    break;
  case Qt::Key_D:
    key_map_[Qt::Key_D] = false;
    break;
  case Qt::Key_Z:
    key_map_[Qt::Key_Z] = false;
    break;
  case Qt::Key_X:
    key_map_[Qt::Key_X] = false;
    break;
  case Qt::Key_Shift:
    key_map_[Qt::Key_Shift] = false;
    break;
  default:
    break;
  }
}
void first_person_interactor::mouse_press_handler(QMouseEvent* event)
{
  last_mouse_position_ = event->pos();
}
void first_person_interactor::mouse_move_handler (QMouseEvent* event)
{
  auto dx = event->x() - last_mouse_position_.x();
  auto dy = event->y() - last_mouse_position_.y();
  if (event->buttons() & Qt::LeftButton)
  {
    transform_->rotate(glm::angleAxis(glm::radians(-look_speed_ * dx), glm::vec3(0.0, 0.0, 1.0)));
    transform_->rotate(glm::angleAxis(glm::radians(-look_speed_ * dy), transform_->right()));
  }

  last_mouse_position_ = event->pos();
}
}
