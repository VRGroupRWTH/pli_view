#ifndef PLI_VIS_INTERACTOR_HPP_
#define PLI_VIS_INTERACTOR_HPP_

class QKeyEvent;
class QMouseEvent;

namespace pli
{
class camera;

class interactor
{
public:
  interactor(camera* camera) : camera_(camera)
  {
    
  }
  virtual ~interactor() = default;

  virtual void update_transform   ()                   { }
  virtual void key_press_handler  (QKeyEvent*   event) { }
  virtual void key_release_handler(QKeyEvent*   event) { }
  virtual void mouse_press_handler(QMouseEvent* event) { }
  virtual void mouse_move_handler (QMouseEvent* event) { }

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

protected:
  camera* camera_;
  float   move_speed_ = 1.0;
  float   look_speed_ = 1.0;
};
}

#endif