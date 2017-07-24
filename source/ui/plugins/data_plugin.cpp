#include <pli_vis/ui/plugins/data_plugin.hpp>

#include <QFileDialog>

#include <pli_vis/io/hdf5_io.hpp>
#include <pli_vis/io/hdf5_io_2.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/utility/line_edit.hpp>
#include <pli_vis/utility/text_browser_sink.hpp>

namespace pli
{
data_plugin::data_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  connect(radio_button_slice_by_slice, &QRadioButton::clicked     , [&]()
  {
    logger_->info(std::string("Toggled sliced (Vervet1818 style) data type."));

    line_edit_file          ->setPlaceholderText("C:/Vervet1818.h5");
    line_edit_vector_spacing->setPlaceholderText("VectorSpacing");
    line_edit_transmittance ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/NTransmittance");
    line_edit_retardation   ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Retardation");
    line_edit_direction     ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Direction");
    line_edit_inclination   ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Inclination");
    line_edit_unit_vector   ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/UnitVector");

    label_direction      ->setEnabled(true );
    label_inclination    ->setEnabled(true );
    label_unit_vector    ->setEnabled(false);

    line_edit_direction  ->setEnabled(true );
    line_edit_inclination->setEnabled(true );
    line_edit_unit_vector->setEnabled(false);

    if (radio_button_slice_by_slice->isChecked())
      set_file(line_edit_file->text().toStdString());
  });
  connect(radio_button_volume        , &QRadioButton::clicked     , [&]()
  {
    logger_->info(std::string("Toggled volumetric (MSA0309 style) data type."));
    
    line_edit_file          ->setPlaceholderText("C:/MSA0309_s0536-0695.h5");
    line_edit_vector_spacing->setPlaceholderText("Voxelsize");
    line_edit_transmittance ->setPlaceholderText("Transmittance");
    line_edit_retardation   ->setPlaceholderText("Retardation");
    line_edit_direction     ->setPlaceholderText("Direction");
    line_edit_inclination   ->setPlaceholderText("Inclination");
    line_edit_unit_vector   ->setPlaceholderText("UnitVector");

    label_direction      ->setEnabled(false);
    label_inclination    ->setEnabled(false);
    label_unit_vector    ->setEnabled(true );

    line_edit_direction  ->setEnabled(false);
    line_edit_inclination->setEnabled(false);
    line_edit_unit_vector->setEnabled(true );

    if (radio_button_volume->isChecked())
      set_file(line_edit_file->text().toStdString());
  });
  connect(button_browse              , &QPushButton::clicked      , [&]
  {
    logger_->info(std::string("Opening file browser."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing file browser. Selection is: {}.", filename.toStdString());
    set_file(filename.toStdString());
  });
  connect(line_edit_vector_spacing   , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit::get_text(line_edit_vector_spacing);
    logger_->info("Vector spacing attribute path is set to {}.", text);
    if (io_)
      io_->set_attribute_path_vector_spacing(text);
    on_change();
  });
  connect(line_edit_transmittance    , &QLineEdit::editingFinished, [&] 
  {
    auto text = line_edit::get_text(line_edit_transmittance);
    logger_->info("Transmittance dataset path is set to {}.", text);
    if (io_)
      io_->set_dataset_path_transmittance(text);
    on_change();
  });
  connect(line_edit_retardation      , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit::get_text(line_edit_retardation);
    logger_->info("Retardation dataset path is set to {}.", text);
    if (io_)
      io_->set_dataset_path_retardation(text);
    on_change();
  });
  connect(line_edit_direction        , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit::get_text(line_edit_direction);
    logger_->info("Fiber direction dataset path is set to {}.", text);
    if (io_)
      io_->set_dataset_path_fiber_direction(text);
    on_change();
  });
  connect(line_edit_inclination      , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit::get_text(line_edit_inclination);
    logger_->info("Fiber inclination dataset path is set to {}.", text);
    if (io_)
      io_->set_dataset_path_fiber_inclination(text);
    on_change();
  });
}
void data_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
  logger_->info(std::string("Start successful."));
}

hdf5_io_base* data_plugin::io() const
{
  return io_.get();
}

void data_plugin::set_file(const std::string& filename)
{
  line_edit_file->setText(filename.c_str());
  io_.reset(nullptr);
  if (filename.empty())
    logger_->info(std::string("Failed to open file: No filepath."));
  else
  {
    if (radio_button_slice_by_slice->isChecked())
      io_.reset(new hdf5_io(
        filename,
        line_edit::get_text(line_edit_vector_spacing),
        ""                                                   ,
        ""                                                   ,
        line_edit::get_text(line_edit_transmittance ),
        line_edit::get_text(line_edit_retardation   ),
        line_edit::get_text(line_edit_direction     ),
        line_edit::get_text(line_edit_inclination   ),
        line_edit::get_text(line_edit_unit_vector   ),
        ""
      ));
    else
      io_.reset(new hdf5_io_2(
        filename,
        line_edit::get_text(line_edit_vector_spacing),
        ""                                                   ,
        ""                                                   ,
        line_edit::get_text(line_edit_transmittance ),
        line_edit::get_text(line_edit_retardation   ),
        line_edit::get_text(line_edit_direction     ),
        line_edit::get_text(line_edit_inclination   ),
        line_edit::get_text(line_edit_unit_vector   ),
        ""
      ));
    logger_->info("Successfully opened file: {}.", filename);
  }
  on_change();
}
}
