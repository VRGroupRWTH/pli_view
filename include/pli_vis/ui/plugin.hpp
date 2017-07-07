#ifndef PLI_VIS_PLUGIN_HPP_
#define PLI_VIS_PLUGIN_HPP_

#include <QWidget>

namespace pli
{
class application;

class plugin : public QWidget
{
public:
  plugin(QWidget* parent = nullptr);

  void set_owner(pli::application* owner);
  
  virtual void awake  ();
  virtual void start  ();
  virtual void destroy();

protected:
  pli::application* owner_ = nullptr;
};
}

#endif
