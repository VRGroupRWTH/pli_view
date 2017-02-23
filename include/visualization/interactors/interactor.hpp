#ifndef PLI_VIS_INTERACTOR_HPP_
#define PLI_VIS_INTERACTOR_HPP_

#include <map>

class QKeyEvent;
class QMouseEvent;

namespace pli
{
class transform;

class interactor
{
public:
  interactor(transform* transform);

  void update_transform();

  void key_press_handler  (QKeyEvent*   event);
  void key_release_handler(QKeyEvent*   event);
  void mouse_move_handler (QMouseEvent* event);

  float move_speed() const
  {
    return move_speed_;
  }
  float look_speed() const
  {
    return look_speed_;
  }

  void set_move_speed(float move_speed)
  {
    move_speed_ = move_speed;
  }
  void set_look_speed(float look_speed)
  {
    look_speed_ = look_speed;
  }

private:
  transform*          transform_  ;
  float               move_speed_ = 1.0;
  float               look_speed_ = 1.0;
  std::map<int, bool> key_map_    ;
};
}

#endif