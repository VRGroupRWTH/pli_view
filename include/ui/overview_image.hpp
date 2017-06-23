#ifndef PLI_VIS_OVERVIEW_IMAGE_HPP_
#define PLI_VIS_OVERVIEW_IMAGE_HPP_

#include <array>
#include <memory>

#include <QLabel>

#include <ui/selection_square.hpp>

namespace pli
{
class overview_image : public QLabel
{
  Q_OBJECT

public:
  overview_image(QWidget* parent = nullptr);

  void set_selection_offset_percentage(const std::array<float, 2>& perc);
  void set_selection_size_percentage  (const std::array<float, 2>& perc);

  std::array<float, 2> selection_offset_percentage() const;
  std::array<float, 2> selection_size_percentage  () const;

signals:
  void on_selection_change(const std::array<float, 2>& offset_perc, const std::array<float, 2>& size_perc);

private:
  void mousePressEvent  (QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent   (QMouseEvent* event) override;

  std::unique_ptr<selection_square> selection_square_;
  bool dragging_ = false;
};
}

#endif
