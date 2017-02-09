#include /* implements */ <ui/plugins/data_plugin.hpp>

#include <limits>

#include <QFileDialog>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
data_plugin::data_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  connect(button_browse_file     , &QPushButton::clicked      , [&]
  {
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing browse dialog. Selection: {}.", filename.toStdString());
    set_file(filename.toStdString());
  });

  connect(line_edit_voxel_size   , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_voxel_size);  
    if (io_)
      io_->set_attribute_path_voxel_size(text);
    logger_->info("Voxel size attribute path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_mask         , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_mask);
    if (io_)
      io_->set_dataset_path_mask(text);
    logger_->info("Mask dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_transmittance, &QLineEdit::editingFinished, [&] 
  {
    auto text = line_edit_utility::get_text(line_edit_transmittance);
    if (io_)
      io_->set_dataset_path_transmittance(text);
    logger_->info("Transmittance dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_retardation  , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_retardation);
    if (io_)
      io_->set_dataset_path_retardation(text);
    logger_->info("Retardation dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_direction    , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_direction);
    if (io_)
      io_->set_dataset_path_fiber_direction(text);
    logger_->info("Fiber direction dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_inclination  , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_inclination);
    if (io_)
      io_->set_dataset_path_fiber_inclination(text);
    logger_->info("Fiber inclination dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(line_edit_distribution , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(line_edit_distribution);
    if (io_)
      io_->set_dataset_path_fiber_distribution(text);
    logger_->info("Fiber distribution dataset path is set to " + text);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });

  connect(checkbox_autoload      , &QCheckBox::stateChanged   , [&] (int state)
  {
    button_load->setEnabled(!state);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
  });
  connect(button_load            , &QPushButton::clicked      , [&]
  {
    on_change(io_.get());
  });
}

hdf5_io<float>* data_plugin::io() const
{
  return io_.get();
}

void data_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));
}

void data_plugin::set_file(const std::string& filename)
{
  line_edit_file->setText(filename.c_str());

  if (filename.empty())
  {
    logger_->info(std::string("Failed to open file. Not specified."));
    io_.reset(nullptr);
    if (checkbox_autoload->isChecked())
      on_change(io_.get());
    return;
  }

  io_.reset(new hdf5_io<float>(
    filename,
    line_edit_utility::get_text(line_edit_voxel_size   ),
    line_edit_utility::get_text(line_edit_mask         ),
    line_edit_utility::get_text(line_edit_transmittance),
    line_edit_utility::get_text(line_edit_retardation  ),
    line_edit_utility::get_text(line_edit_direction    ),
    line_edit_utility::get_text(line_edit_inclination  ),
    line_edit_utility::get_text(line_edit_distribution )
  ));

  logger_->info("Opened file: " + filename);

  if (checkbox_autoload->isChecked())
    on_change(io_.get());
}
}
