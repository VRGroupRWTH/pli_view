/* Original Work Copyright (c) 2012-2014 Alexander Turkin
Modified 2014 by William Hallatt
Modified 2015 by Jacob Dawid
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <QWidget>
#include <QTimer>
#include <QColor>

namespace pli
{
class wait_spinner : public QWidget 
{
  Q_OBJECT
public:
  wait_spinner(                             QWidget* parent = nullptr, bool center = true, bool disable_parent = true);
  wait_spinner(Qt::WindowModality modality, QWidget* parent = nullptr, bool center = true, bool disable_parent = true);

  wait_spinner(const wait_spinner&) = delete;
  wait_spinner& operator=(const wait_spinner&) = delete;

  void start();
  void stop ();

  void set_color                 (QColor  color                 );
  void set_roundness             (qreal   roundness             );
  void set_minimum_trail_opacity (qreal   minimum_trail_opacity );
  void set_trail_fade (qreal   trail                 );
  void set_revolutions_per_second(qreal   revolutions_per_second);
  void set_number_of_lines       (int     lines                 );
  void set_line_length           (int     length                );
  void set_line_width            (int     width                 );
  void set_inner_radius          (int     radius                );
  void set_text                  (QString text                  );

  QColor color                  () const;
  qreal  roundness              () const;
  qreal  minimum_trail_opacity  () const;
  qreal  trail_fade  () const;
  qreal  revolutions_pers_second() const;
  int    number_of_lines        () const;
  int    line_length            () const;
  int    line_width             () const;
  int    inner_radius           () const;

  bool   is_spinning            () const;

private slots:
  void rotate();

protected:
  void paintEvent(QPaintEvent* paint_event) override;

  static int    line_count_distance(int current, int primary, int total_line_count);
  static QColor calculate_color    (int distance, int total_line_count, qreal trail_fade, qreal min_opacity, QColor color);

  void initialize     ();
  void update_size    ();
  void update_timer   ();
  void update_position();
  
  bool    center_                 ;
  bool    disable_parent_         ;

  QColor  color_                  = QColor(255, 255, 255);
  qreal   roundness_              = 75.0;
  qreal   minimum_trail_opacity_  = 25.0;
  qreal   trail_fade_             = 75.0;
  qreal   revolutions_per_second_ = 1   ;
  int     number_of_lines_        = 24  ;
  int     line_length_            = 32  ;
  int     line_width_             = 8   ;
  int     inner_radius_           = 24  ;

  QTimer* timer_                  ;
  int     current_counter_        = 0;
  bool    is_spinning_            = false;
};
}