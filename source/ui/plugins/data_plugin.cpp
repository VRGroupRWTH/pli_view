#include /* implements */ <ui/plugins/data_plugin.hpp>

#include <QFileDialog>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
data_plugin::data_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  connect(radio_button_sliced     , &QRadioButton::clicked     , [&]()
  {
    logger_->info(std::string("Toggled sliced (Vervet1818 style) data type."));
    if (radio_button_sliced->isChecked())
      set_file(line_edit_file->text().toStdString());
  });
  connect(radio_button_volumetric , &QRadioButton::clicked     , [&]()
  {
    logger_->info(std::string("Toggled volumetric (MSA0309 style) data type."));
    if (radio_button_volumetric->isChecked())
      set_file(line_edit_file->text().toStdString());
  });

  connect(button_browse_file      , &QPushButton::clicked      , [&]
  {
    logger_->info(std::string("Opening file browser."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing file browser. Selection is: {}.", filename.toStdString());
    set_file(filename.toStdString());
  });

  connect(line_edit_vector_spacing, &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_vector_spacing);
    logger_->info("Vector spacing attribute path is set to {}." + text);
    if (io_)
    {
      io_->set_attribute_path_vector_spacing(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_block_size    , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_block_size);
    logger_->info("Block size attribute path is set to {}." + text);
    if (io_)
    {
      io_->set_attribute_path_block_size(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_mask          , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_mask);
    logger_->info("Mask dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_mask(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_transmittance , &QLineEdit::editingFinished, [&] 
  {
    auto text = line_edit_utility::get_text(line_edit_transmittance);
    logger_->info("Transmittance dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_transmittance(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_retardation   , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_retardation);
    logger_->info("Retardation dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_retardation(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_direction     , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_direction);
    logger_->info("Fiber direction dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_fiber_direction(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_inclination   , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_inclination);
    logger_->info("Fiber inclination dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_fiber_inclination(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });
  connect(line_edit_distribution  , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_distribution);
    logger_->info("Fiber distribution dataset path is set to {}." + text);
    if (io_)
    {
      io_->set_dataset_path_fiber_distribution(text);
      if (checkbox_autoload->isChecked())
        on_change(io_.get());
    }
  });

  connect(checkbox_autoload       , &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info("Auto load is {}.", state ? "enabled" : "disabled");
    button_load->setEnabled(!state);
    if (io_ != nullptr && checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(button_load             , &QPushButton::clicked      , [&]
  {
    if (io_ != nullptr)
      on_change(io_.get());
  });
}

hdf5_io_base* data_plugin::io() const
{
  return io_.get();
}

void data_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  logger_->info(std::string("Start successful."));
}

void data_plugin::set_file(const std::string& filename)
{
  io_.reset(nullptr);

  line_edit_file->setText(filename.c_str());

  if (filename.empty())
  {
    logger_->info(std::string("Failed to open file: No filepath."));
    return;
  }

  if (radio_button_sliced->isChecked())
    io_.reset(new hdf5_io(
      filename,
      line_edit_utility::get_text(line_edit_vector_spacing),
      line_edit_utility::get_text(line_edit_block_size    ),
      line_edit_utility::get_text(line_edit_mask          ),
      line_edit_utility::get_text(line_edit_transmittance ),
      line_edit_utility::get_text(line_edit_retardation   ),
      line_edit_utility::get_text(line_edit_direction     ),
      line_edit_utility::get_text(line_edit_inclination   ),
      line_edit_utility::get_text(line_edit_distribution  )
    ));
  else
    io_.reset(new hdf5_io_2(
      filename,
      line_edit_utility::get_text(line_edit_vector_spacing),
      line_edit_utility::get_text(line_edit_block_size    ),
      line_edit_utility::get_text(line_edit_mask          ),
      line_edit_utility::get_text(line_edit_transmittance ),
      line_edit_utility::get_text(line_edit_retardation   ),
      line_edit_utility::get_text(line_edit_direction     ),
      line_edit_utility::get_text(line_edit_inclination   ),
      line_edit_utility::get_text(line_edit_distribution  )
    ));

  logger_->info("Successfully opened file: {}.", filename);

  if (checkbox_autoload->isChecked())
    on_change(io_.get());
}
}
