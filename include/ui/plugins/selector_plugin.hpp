#ifndef PLI_VIS_SELECTOR_HPP_
#define PLI_VIS_SELECTOR_HPP_

#include <array>
#include <cstddef>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_selector_toolbox.h>

namespace pli
{
class window;

class selector_plugin : public plugin, public Ui_selector_toolbox, public loggable<selector_plugin>
{
  Q_OBJECT
public:
  selector_plugin(QWidget* parent = nullptr);
  
  std::array<std::size_t, 3> selection_offset() const;
  std::array<std::size_t, 3> selection_size  () const;

  void start() override;

signals:
  void on_change(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size);

private:
  QSize sizeHint   () const override
  {
    auto s     = size ();
    last_width = width();
    s.setWidth (QWidget::sizeHint().width());
    s.setHeight(image->pixmap() ? static_cast<float>(width()) * image->pixmap()->height() / image->pixmap()->width() : height());
    return s;
  }
  void  resizeEvent(QResizeEvent * event) override
  {
    QWidget::resizeEvent(event);
    if (last_width != width())
    {
      updateGeometry();
      update();
    }
  }

  void upload();

  mutable int last_width;
};
}

#endif
