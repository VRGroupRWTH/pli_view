#ifndef PLI_VIS_SPAN_SLIDER_HPP
#define PLI_VIS_SPAN_SLIDER_HPP

#include <QObject>
#include <QSlider>
#include <QStyle>

class QStylePainter;
class QStyleOptionSlider;

class span_slider : public QSlider
{
  Q_OBJECT
public:
  explicit span_slider(                             QWidget* parent = nullptr);
  explicit span_slider(Qt::Orientation orientation, QWidget* parent = nullptr);
  virtual ~span_slider() { }

  enum handle_movement_mode
  {
    FreeMovement ,
    NoCrossing   ,
    NoOverlapping
  };
  enum span_handle
  {
    NoHandle   ,
    LowerHandle,
    UpperHandle
  };

  handle_movement_mode movement_mode     () const;
  void                 set_movement_mode (handle_movement_mode mode);
  int                  lower_value       () const;
  void                 set_lower_value   (int lower);
  int                  upper_value       () const;
  void                 set_upper_value   (int upper);
  int                  lower_position    () const;
  void                 set_lower_position(int lower);
  int                  upper_position    () const;
  void                 set_upper_position(int upper);

public slots:
  void set_span            (int lower, int upper);
  void update_range        (int min, int max);
  void move_pressed_handle ();

protected:
  void keyPressEvent       (QKeyEvent*   event) override;
  void mousePressEvent     (QMouseEvent* event) override;
  void mouseMoveEvent      (QMouseEvent* event) override;
  void mouseReleaseEvent   (QMouseEvent* event) override;
  void paintEvent          (QPaintEvent* event) override;
  
  void init_style_option       (QStyleOptionSlider* option  , span_handle         handle = UpperHandle) const;
  int  pick                    (const QPoint&       point   ) const;
  int  pixel_pos_to_range_value(int                 position) const;
  void handle_mouse_press      (const QPoint&       position, QStyle::SubControl& control, int value, span_handle handle);
  void setup_painter           (QPainter*           painter , Qt::Orientation     orientation, qreal x1, qreal y1, qreal x2, qreal y2) const;
  void draw_span               (QStylePainter*      painter , const QRect&        rectangle) const;
  void draw_handle             (QStylePainter*      painter , span_handle         handle   ) const;
  void trigger_action          (SliderAction        action  , bool                main     );
  void swap_controls           ();
  
  void span_changed          (int lower, int upper);
  void lower_value_changed   (int lower);
  void upper_value_changed   (int upper);
  void lower_position_changed(int lower);
  void upper_position_changed(int upper);
  void slider_pressed        (span_handle handle);

  bool                 first_movement_;
  bool                 block_tracking_;
  int                  lower_         ;
  int                  upper_         ;
  int                  lower_position_;
  int                  upper_position_;
  int                  offset_        ;
  int                  position_      ;
  span_handle          last_pressed_  ;
  span_handle          main_control_  ;
  handle_movement_mode movement_mode_ ;
  QStyle::SubControl   lower_pressed_ ;
  QStyle::SubControl   upper_pressed_ ;
};

#endif
