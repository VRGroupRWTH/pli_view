#include <pli_vis/ui/plugins/demo_plugin.hpp>

#include <boost/lexical_cast.hpp>
#include <json/json.hpp>

#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
demo_plugin::demo_plugin        (QWidget* parent) : plugin(parent), buttons_{button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8, button_9, button_10}
{
  if (!std::ifstream(presets_filepath_).good())
    create_default();
  
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

void demo_plugin::start         ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
}

void demo_plugin::create_default() const
{
  nlohmann::json json;
  json["presets"] = nlohmann::json::array({
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object(), 
    nlohmann::json::object()});

  std::ofstream file(presets_filepath_);
  file << std::setw(4) << json << std::endl;
}
void demo_plugin::load_preset   (std::size_t index) const
{
  std::ifstream  file(presets_filepath_);
  nlohmann::json json;
  file >> json;

  // TODO: Read current state from preset i in json.
}
void demo_plugin::save_preset   (std::size_t index) const
{
  std::ifstream  file(presets_filepath_);
  nlohmann::json json;
  file >> json;

  // TODO: Write current state to preset i in json.

  std::ofstream mutable_file(presets_filepath_);
  mutable_file << std::setw(4) << json << std::endl;
}
}
