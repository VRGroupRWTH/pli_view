#include <pli_vis/ui/plugins/polar_plot_plugin.hpp>

#include <pli_vis/cuda/polar_plot.h>
#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
polar_plot_plugin::polar_plot_plugin(QWidget* parent)
{
  line_edit_superpixel_size   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_angular_partitions->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_superpixel_size   ->setText     (QString::fromStdString(std::to_string(slider_superpixel_size   ->value())));
  line_edit_angular_partitions->setText     (QString::fromStdString(std::to_string(slider_angular_partitions->value())));

  connect(checkbox_enabled            , &QCheckBox::stateChanged    , [&](int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    field_->set_active(state);
  });
  connect(slider_superpixel_size      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_superpixel_size->setText(QString::fromStdString(std::to_string(slider_superpixel_size->value())));
  });
  connect(line_edit_superpixel_size   , &QLineEdit::editingFinished , [&]
  {
    slider_superpixel_size->setValue(line_edit::get_text<int>(line_edit_superpixel_size));
  });
  connect(slider_angular_partitions   , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_angular_partitions->setText(QString::fromStdString(std::to_string(slider_angular_partitions->value())));
  });
  connect(line_edit_angular_partitions, &QLineEdit::editingFinished , [&]
  {
    slider_angular_partitions->setValue(line_edit::get_text<int>(line_edit_angular_partitions));
  });
  connect(button_calculate            , &QAbstractButton::clicked   , [&]
  {
    owner_ ->set_is_loading(true);
    logger_->info(std::string("Updating viewer..."));

    auto vectors            = owner_->get_plugin<data_plugin>()->generate_vectors(false);
    auto superpixel_size    = line_edit::get_text<unsigned>(line_edit_superpixel_size   );
    auto angular_partitions = line_edit::get_text<unsigned>(line_edit_angular_partitions);
    auto symmetric          = checkbox_symmetric->isChecked();
    
    std::array<std::vector<float3>, 2> polar_plots;
    future_ = std::async(std::launch::async, [&]
    {
      try
      {
        auto vectors_dimensions = make_uint2(vectors.shape()[0] / superpixel_size, vectors.shape()[1] / superpixel_size) * superpixel_size;
        vectors.resize(boost::extents[vectors_dimensions.x][vectors_dimensions.y][1]);
        polar_plots = calculate(
          std::vector<float3>(vectors.data(), vectors.data() + vectors.num_elements()), 
          vectors_dimensions,
          superpixel_size   , 
          angular_partitions, 
          symmetric         );
      }
      catch (std::exception& exception) { logger_->error(std::string(exception.what())); }
    });
    while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      QApplication::processEvents();
    
    field_->set_data(polar_plots[0], polar_plots[1]);

    logger_->info(std::string("Update successful."));
    owner_ ->set_is_loading(false);
  });
}

void polar_plot_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  field_ = owner_->viewer->add_renderable<polar_plot_field>();
  connect(owner_->get_plugin<color_plugin>(), &color_plugin::on_change, [&] (int mode, float k, bool inverted)
  {
    field_->set_color_mapping(mode, k, inverted);
  });
}
}
