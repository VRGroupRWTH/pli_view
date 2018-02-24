#include <pli_vis/ui/widgets/remote_viewer.hpp>

namespace pli
{
remote_viewer::remote_viewer(QWidget* parent) : QLabel(parent)
{
  QImage image;
  image.load("reply.jpg");
  setPixmap (QPixmap::fromImage(image));
  show      ();
}
}
