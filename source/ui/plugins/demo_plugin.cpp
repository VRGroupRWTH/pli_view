#include <pli_vis/ui/plugins/demo_plugin.hpp>

#include <boost/lexical_cast.hpp>

#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
demo_plugin::demo_plugin(QWidget* parent) : plugin(parent), buttons_{button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8, button_9, button_10}
{
  for (auto& button : buttons_)
  {
    connect(button, &QPushButton::clicked, [&]
    {
      bool save  = QApplication::keyboardModifiers() & Qt::ControlModifier;
      auto index = boost::lexical_cast<std::size_t>(button->text().toStdString());
      save ? save_preset(index) : load_preset(index);
      logger_->info("{} preset {}.", save ? "Saved" : "Loaded", index);
    });
  }
}

void demo_plugin::start      ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
}
void demo_plugin::load_preset(std::size_t index)
{

}
void demo_plugin::save_preset(std::size_t index)
{

}
}
