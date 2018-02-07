#ifndef PLI_VIS_PLUGIN_BASE_HPP_
#define PLI_VIS_PLUGIN_BASE_HPP_

#include <QWidget>

namespace pli
{
class application;

class plugin_base : public QWidget
{
public:
  explicit plugin_base(QWidget* parent = nullptr) : QWidget(parent) { }
  virtual ~plugin_base() = default;

  virtual void set_owner(application* owner) = 0;
  virtual void awake    ()                   = 0;
  virtual void start    ()                   = 0;
  virtual void destroy  ()                   = 0;
};
}

#endif
