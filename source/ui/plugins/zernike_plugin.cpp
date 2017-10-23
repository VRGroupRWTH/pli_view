#include <pli_vis/ui/plugins/zernike_plugin.hpp>

#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
zernike_plugin::zernike_plugin(QWidget* parent)
{
  line_edit_superpixel_x    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_superpixel_y    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_partitions_theta->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_partitions_rho  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_maximum_degree  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  line_edit_superpixel_x    ->setText(QString::fromStdString(std::to_string(slider_superpixel_x    ->value())));
  line_edit_superpixel_y    ->setText(QString::fromStdString(std::to_string(slider_superpixel_y    ->value())));
  line_edit_partitions_theta->setText(QString::fromStdString(std::to_string(slider_partitions_theta->value())));
  line_edit_partitions_rho  ->setText(QString::fromStdString(std::to_string(slider_partitions_rho  ->value())));
  line_edit_maximum_degree  ->setText(QString::fromStdString(std::to_string(slider_maximum_degree  ->value())));

  connect(checkbox_enabled          , &QCheckBox::stateChanged    , [&](int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    //tensor_field_->set_active(state);
  });
                                    
  connect(slider_superpixel_x       , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_superpixel_x    ->setText(QString::fromStdString(std::to_string(slider_superpixel_x    ->value())));
  });
  connect(slider_superpixel_y       , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_superpixel_y    ->setText(QString::fromStdString(std::to_string(slider_superpixel_y    ->value())));
  });
  connect(slider_partitions_theta   , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_partitions_theta->setText(QString::fromStdString(std::to_string(slider_partitions_theta->value())));
  });
  connect(slider_partitions_rho     , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_partitions_rho  ->setText(QString::fromStdString(std::to_string(slider_partitions_rho  ->value())));
  });
  connect(slider_maximum_degree     , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_maximum_degree  ->setText(QString::fromStdString(std::to_string(slider_maximum_degree  ->value())));
  });

  connect(line_edit_superpixel_x    , &QLineEdit::editingFinished , [&]
  {
    slider_superpixel_x    ->setValue(line_edit::get_text<int>(line_edit_superpixel_x));
  });
  connect(line_edit_superpixel_y    , &QLineEdit::editingFinished , [&]
  {
    slider_superpixel_y    ->setValue(line_edit::get_text<int>(line_edit_superpixel_y));
  });
  connect(line_edit_partitions_theta, &QLineEdit::editingFinished , [&]
  {
    slider_partitions_theta->setValue(line_edit::get_text<int>(line_edit_partitions_theta));
  });
  connect(line_edit_partitions_rho  , &QLineEdit::editingFinished , [&]
  {
    slider_partitions_rho  ->setValue(line_edit::get_text<int>(line_edit_partitions_rho));
  });
  connect(line_edit_maximum_degree  , &QLineEdit::editingFinished , [&]
  {
    slider_maximum_degree  ->setValue(line_edit::get_text<int>(line_edit_maximum_degree));
  });

  connect(button_calculate          , &QAbstractButton::clicked   , [&]
  {
    owner_ ->set_is_loading(true);
    logger_->info(std::string("Updating viewer..."));

    future_ = std::async(std::launch::async, [&]
    {
      try
      {
        auto  parameters            = get_parameters();
        auto  vectors               = owner_->get_plugin<data_plugin>()->generate_vectors(false);
        uint2 superpixel_dimensions = 
        {
          unsigned(vectors.shape()[0]) / parameters.superpixel_size.x,
          unsigned(vectors.shape()[1]) / parameters.superpixel_size.y,
        };
        auto  vector_dimensions     = parameters.superpixel_size * superpixel_dimensions;
        vectors.resize(boost::extents[vector_dimensions.x][vector_dimensions.y][1]);

        //... = zer::launch(
        //  vectors                   , 
        //  parameters.superpixel_size, 
        //  parameters.partitions     , 
        //  parameters.maximum_degree , 
        //  parameters.symmetric      );
      }
      catch (std::exception& exception)
      {
        logger_->error(std::string(exception.what()));
      }
    });
    while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      QApplication::processEvents();

    //tensor_field_->set_data(...);

    logger_->info(std::string("Update successful."));
    owner_ ->set_is_loading(false);
  });
}

void                       zernike_plugin::start         ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  //tensor_field_ = owner_->viewer->add_renderable<tensor_field>();
  //connect(owner_->get_plugin<color_plugin>(), &color_plugin::on_change, [&](int mode, float k, bool inverted)
  //{
  //  tensor_field_->set_color_mapping(mode, k, inverted);
  //});
}
zernike_plugin::parameters zernike_plugin::get_parameters() const
{
  return 
  {
    {
      line_edit::get_text<unsigned>(line_edit_superpixel_x),
      line_edit::get_text<unsigned>(line_edit_superpixel_y) 
    },
    {
      line_edit::get_text<unsigned>(line_edit_partitions_theta),
      line_edit::get_text<unsigned>(line_edit_partitions_rho  ) 
    },
    line_edit::get_text<unsigned>(line_edit_maximum_degree),
    checkbox_symmetric->isChecked()
  };
}
}
