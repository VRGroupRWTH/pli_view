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
      save ? save_preset(index - 1) : load_preset(index - 1);
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

  auto  data_plugin                    = owner_->get_plugin<pli::data_plugin>              ();
  auto  interactor_plugin              = owner_->get_plugin<pli::interactor_plugin>        ();
  auto  color_plugin                   = owner_->get_plugin<pli::color_plugin>             ();
  auto  scalar_plugin                  = owner_->get_plugin<pli::scalar_plugin>            ();
  auto  fom_plugin                     = owner_->get_plugin<pli::fom_plugin>               ();
  auto  polar_plot_plugin              = owner_->get_plugin<pli::polar_plot_plugin>        ();
  auto  odf_plugin                     = owner_->get_plugin<pli::odf_plugin>               ();
  auto  local_tractography_plugin      = owner_->get_plugin<pli::local_tractography_plugin>();

  auto& preset                         = json["presets"][index];
  auto& data_plugin_data               = preset["data_plugin"              ];
  auto& interactor_plugin_data         = preset["interactor_plugin"        ];
  auto& color_plugin_data              = preset["color_plugin"             ];
  auto& scalar_plugin_data             = preset["scalar_plugin"            ];
  auto& fom_plugin_data                = preset["fom_plugin"               ];
  auto& polar_plot_plugin_data         = preset["polar_plot_plugin"        ];
  auto& odf_plugin_data                = preset["odf_plugin"               ];
  auto& local_tractography_plugin_data = preset["local_tractography_plugin"];

  // TODO: Read current state from json.
}
void demo_plugin::save_preset   (std::size_t index) const
{
  std::ifstream  file(presets_filepath_);
  nlohmann::json json;
  file >> json;
  
  auto  data_plugin                    = owner_->get_plugin<pli::data_plugin>              ();
  auto  interactor_plugin              = owner_->get_plugin<pli::interactor_plugin>        ();
  auto  color_plugin                   = owner_->get_plugin<pli::color_plugin>             ();
  auto  scalar_plugin                  = owner_->get_plugin<pli::scalar_plugin>            ();
  auto  fom_plugin                     = owner_->get_plugin<pli::fom_plugin>               ();
  auto  polar_plot_plugin              = owner_->get_plugin<pli::polar_plot_plugin>        ();
  auto  odf_plugin                     = owner_->get_plugin<pli::odf_plugin>               ();
  auto  local_tractography_plugin      = owner_->get_plugin<pli::local_tractography_plugin>();

  auto& preset                         = json["presets"][index];
  preset["data_plugin"              ]  = nlohmann::json::object();
  preset["interactor_plugin"        ]  = nlohmann::json::object();
  preset["color_plugin"             ]  = nlohmann::json::object();
  preset["scalar_plugin"            ]  = nlohmann::json::object();
  preset["fom_plugin"               ]  = nlohmann::json::object();
  preset["polar_plot_plugin"        ]  = nlohmann::json::object();
  preset["odf_plugin"               ]  = nlohmann::json::object();
  preset["local_tractography_plugin"]  = nlohmann::json::object();
  auto& data_plugin_data               = preset["data_plugin"              ];
  auto& interactor_plugin_data         = preset["interactor_plugin"        ];
  auto& color_plugin_data              = preset["color_plugin"             ];
  auto& scalar_plugin_data             = preset["scalar_plugin"            ];
  auto& fom_plugin_data                = preset["fom_plugin"               ];
  auto& polar_plot_plugin_data         = preset["polar_plot_plugin"        ];
  auto& odf_plugin_data                = preset["odf_plugin"               ];
  auto& local_tractography_plugin_data = preset["local_tractography_plugin"];
  // TODO: Write current state to json.

  std::ofstream mutable_file(presets_filepath_);
  mutable_file << std::setw(4) << json << std::endl;
}
}
