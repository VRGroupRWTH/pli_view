#include <pli_vis/ui/plugins/scalar_plugin.hpp>

#include <vector_functions.hpp>

#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/algorithms/scalar_field.hpp>

namespace pli
{
scalar_plugin::scalar_plugin(QWidget* parent) : plugin(parent)
{
  connect(checkbox_enabled      , &QCheckBox::stateChanged, [&](bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    scalar_field_->set_active(state);
  });
  connect(checkbox_transmittance, &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Transmittance maps are selected."));
    upload();
  });
  connect(checkbox_retardation  , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Retardation maps are selected."));
    upload();
  });
  connect(checkbox_direction    , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Direction maps are selected."));
    upload();
  });
  connect(checkbox_inclination  , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Inclination maps are selected."));
    upload();
  });
}

void scalar_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  scalar_field_ = owner_->viewer->add_renderable<scalar_field>();

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_load, [&]
  {
    upload();
  });
}
void scalar_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto data_plugin = owner_->get_plugin<pli::data_plugin>();

  if (checkbox_transmittance->isChecked())
  {
    auto& data    = data_plugin->transmittance();
    auto  subdata = boost::multi_array<float, 2>(data[boost::indices[boost::multi_array_types::index_range()][boost::multi_array_types::index_range()][data.shape()[2] - 1]]);
    scalar_field_->set_data       (make_uint3(subdata.shape()[0], subdata.shape()[1], 1), subdata.data());
    scalar_field_->set_translation(glm::vec3(0, 0, data.shape()[2]));
  }
  if (checkbox_retardation  ->isChecked())
  {
    auto data    = data_plugin->retardation();
    auto subdata = boost::multi_array<float, 2>(data[boost::indices[boost::multi_array_types::index_range()][boost::multi_array_types::index_range()][data.shape()[2] - 1]]);
    scalar_field_->set_data       (make_uint3(subdata.shape()[0], subdata.shape()[1], 1), subdata.data());
    scalar_field_->set_translation(glm::vec3(0, 0, data.shape()[2]));
  }
  if (checkbox_direction    ->isChecked())
  {
    auto data    = data_plugin->direction();
    auto subdata = boost::multi_array<float, 2>(data[boost::indices[boost::multi_array_types::index_range()][boost::multi_array_types::index_range()][data.shape()[2] - 1]]);
    scalar_field_->set_data       (make_uint3(subdata.shape()[0], subdata.shape()[1], 1), subdata.data());
    scalar_field_->set_translation(glm::vec3(0, 0, data.shape()[2]));
  }
  if (checkbox_inclination  ->isChecked())
  {
    auto data    = data_plugin->inclination();
    auto subdata = boost::multi_array<float, 2>(data[boost::indices[boost::multi_array_types::index_range()][boost::multi_array_types::index_range()][data.shape()[2] - 1]]);
    scalar_field_->set_data       (make_uint3(subdata.shape()[0], subdata.shape()[1], 1), subdata.data());
    scalar_field_->set_translation(glm::vec3(0, 0, data.shape()[2]));
  }

  logger_->info(std::string("Update successful."));
}
}
