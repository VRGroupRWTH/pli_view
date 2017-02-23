#include /* implements */ <visualization/interactors/interactor.hpp>

#include <QKeyEvent>

#include <math/transform.hpp>

namespace pli
{
interactor::interactor(transform* transform) : transform_(transform)
{

}

void interactor::update_transform()
{
  if (key_map_[Qt::Key_W])
    transform_->translate({           0,  move_speed_, 0});
  if (key_map_[Qt::Key_A])
    transform_->translate({-move_speed_,            0, 0});
  if (key_map_[Qt::Key_S])
    transform_->translate({           0, -move_speed_, 0});
  if (key_map_[Qt::Key_D])
    transform_->translate({ move_speed_,            0, 0 });
  if (key_map_[Qt::Key_Z])
    transform_->translate({           0,            0,  move_speed_});
  if (key_map_[Qt::Key_X])
    transform_->translate({           0,            0, -move_speed_});
}

void interactor::key_press_handler  (QKeyEvent*   event)
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
  default: 
    break;
  }
}
void interactor::key_release_handler(QKeyEvent*   event)
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
  default:
    break;
  }
}
void interactor::mouse_move_handler (QMouseEvent* event)
{

}
}