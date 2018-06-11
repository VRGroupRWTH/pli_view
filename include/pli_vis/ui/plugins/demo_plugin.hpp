#ifndef PLI_VIS_DEMO_PLUGIN_HPP_
#define PLI_VIS_DEMO_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_demo_toolbox.h>

namespace pli
{
class demo_plugin : public plugin<demo_plugin, Ui_demo_toolbox>
{
public:
  explicit demo_plugin  (QWidget* parent = nullptr);
  demo_plugin           (const demo_plugin&  that) = delete ;
  demo_plugin           (      demo_plugin&& temp) = default;
  virtual ~demo_plugin   ()                        = default;
  demo_plugin& operator=(const demo_plugin&  that) = delete ;
  demo_plugin& operator=(      demo_plugin&& temp) = default;

protected:
  void start         () override;

  void create_default() const;
  void load_preset   (std::size_t index) const;
  void save_preset   (std::size_t index) const;

  std::vector<QPushButton*> buttons_          ;
  std::string               presets_filepath_ = "presets.json";
};
}

#endif