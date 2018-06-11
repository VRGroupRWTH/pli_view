#include <pli_vis/ui/plugins/demo_plugin.hpp>

#include <boost/lexical_cast.hpp>
#include <json/json.hpp>

#include <pli_vis/ui/utility/line_edit.hpp>
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

  data_plugin_data      ["dataset"            ] = data_plugin->filepath        ();
  data_plugin_data      ["offset"             ] = data_plugin->selection_offset();
  data_plugin_data      ["size"               ] = data_plugin->selection_size  ();
  data_plugin_data      ["stride"             ] = data_plugin->selection_stride();
  
  auto translation = owner_->viewer->camera()->translation   ();
  auto rotation    = owner_->viewer->camera()->rotation_euler();
  interactor_plugin_data["translation"        ] = {translation.x, translation.y, translation.z};
  interactor_plugin_data["rotation"           ] = {rotation   .x, rotation   .y, rotation   .z};
                                              
  color_plugin_data     ["mode"               ] = std::uint32_t(color_plugin->mode());
  color_plugin_data     ["k"                  ] = color_plugin->k       ();
  color_plugin_data     ["invert_p"           ] = color_plugin->inverted();
                                              
  scalar_plugin_data    ["enabled"            ] = scalar_plugin->checkbox_enabled    ->isChecked();
  scalar_plugin_data    ["mode"               ] = scalar_plugin->checkbox_retardation->isChecked(); // 0 - transmittance, 1 - retardation
                                              
  fom_plugin_data       ["enabled"            ] = fom_plugin->checkbox_enabled->isChecked();
  fom_plugin_data       ["scale"              ] = line_edit::get_text<float>(fom_plugin->line_edit_fiber_scale);
                                              
  polar_plot_plugin_data["enabled"            ] = polar_plot_plugin->checkbox_enabled  ->isChecked();
  polar_plot_plugin_data["symmetric"          ] = polar_plot_plugin->checkbox_symmetric->isChecked();
  polar_plot_plugin_data["superpixel_size"    ] = line_edit::get_text<std::size_t>(polar_plot_plugin->line_edit_superpixel_size);
  polar_plot_plugin_data["angular_partitions" ] = line_edit::get_text<std::size_t>(polar_plot_plugin->line_edit_angular_partitions);
                                              
  odf_plugin_data       ["enabled"            ] = odf_plugin->checkbox_enabled  ->isChecked();
  odf_plugin_data       ["symmetric"          ] = odf_plugin->checkbox_even_only->isChecked();
  odf_plugin_data       ["supervoxel_extent"  ] = {
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_vector_block_x), 
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_vector_block_y), 
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_vector_block_z)};
  odf_plugin_data       ["histogram_bins"     ] = {
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_histogram_theta),
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_histogram_phi  )};
  odf_plugin_data       ["maximum_sh_degree"  ] = 
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_maximum_sh_degree);
  odf_plugin_data       ["sampling_partitions"] = {
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_sampling_theta),
    line_edit::get_text<std::size_t>(odf_plugin->line_edit_sampling_phi  )};
  odf_plugin_data       ["hierarchical"       ] = odf_plugin->checkbox_hierarchical->isChecked();
  odf_plugin_data       ["visible_layers"     ] = {
    odf_plugin->checkbox_depth_0->isChecked(),
    odf_plugin->checkbox_depth_1->isChecked(),
    odf_plugin->checkbox_depth_2->isChecked(),
    odf_plugin->checkbox_depth_3->isChecked(),
    odf_plugin->checkbox_depth_4->isChecked(),
    odf_plugin->checkbox_depth_5->isChecked(),
    odf_plugin->checkbox_depth_6->isChecked(),
    odf_plugin->checkbox_depth_7->isChecked(),
    odf_plugin->checkbox_depth_8->isChecked(),
    odf_plugin->checkbox_depth_9->isChecked()};

  local_tractography_plugin_data["enabled"          ] = local_tractography_plugin->checkbox_enabled->isChecked();
  local_tractography_plugin_data["offset"           ] = local_tractography_plugin->seed_offset      ();
  local_tractography_plugin_data["size"             ] = local_tractography_plugin->seed_size        ();
  local_tractography_plugin_data["stride"           ] = local_tractography_plugin->seed_stride      ();
  local_tractography_plugin_data["integration_step" ] = local_tractography_plugin->step             ();
  local_tractography_plugin_data["iterations"       ] = local_tractography_plugin->iterations       ();
  local_tractography_plugin_data["streamline_radius"] = local_tractography_plugin->streamline_radius();
  local_tractography_plugin_data["remote_address"   ] = line_edit::get_text<std::string>(local_tractography_plugin->line_edit_remote_address);
  local_tractography_plugin_data["remote_folder"    ] = line_edit::get_text<std::string>(local_tractography_plugin->line_edit_dataset_folder);

  std::ofstream mutable_file(presets_filepath_);
  mutable_file << std::setw(4) << json << std::endl;
}
}
