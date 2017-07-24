#include <pli_vis/ui/plugins/data_plugin.hpp>

#include <QFileDialog>

#include <pli_vis/io/io_slice_impl.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
data_plugin::data_plugin(QWidget* parent) 
: plugin(parent), 
  io_(line_edit::get_text(line_edit_file          ), 
      line_edit::get_text(line_edit_vector_spacing),
      line_edit::get_text(line_edit_transmittance ),
      line_edit::get_text(line_edit_retardation   ),
      line_edit::get_text(line_edit_direction     ),
      line_edit::get_text(line_edit_inclination   ),
      line_edit::get_text(line_edit_mask          ),
      line_edit::get_text(line_edit_unit_vector   ))
{
  setupUi(this);
  
  connect(button_browse              , &QPushButton::clicked      , [&]
  {
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)")).toStdString();
    
    io_.set_filepath       (filename.c_str());
    line_edit_file->setText(filename.c_str());

    filename.empty() 
      ? logger_->info(std::string("Failed to open file: No filepath.")) 
      : logger_->info("Successfully opened file: {}.", filename);

    on_change();
  });
  connect(line_edit_vector_spacing   , &QLineEdit::editingFinished, [&]
  {
    io_.set_vector_spacing_path(line_edit::get_text(line_edit_vector_spacing));
    on_change();
  });
  connect(line_edit_transmittance    , &QLineEdit::editingFinished, [&] 
  {
    io_.set_transmittance_path(line_edit::get_text(line_edit_transmittance));
    on_change();
  });
  connect(line_edit_retardation      , &QLineEdit::editingFinished, [&]
  {
    io_.set_retardation_path(line_edit::get_text(line_edit_retardation));
    on_change();
  });
  connect(line_edit_direction        , &QLineEdit::editingFinished, [&]
  {
    io_.set_direction_path(line_edit::get_text(line_edit_direction));
    on_change();
  });
  connect(line_edit_inclination      , &QLineEdit::editingFinished, [&]
  {
    io_.set_inclination_path(line_edit::get_text(line_edit_inclination));
    on_change();
  });
  connect(line_edit_mask             , &QLineEdit::editingFinished, [&]
  {
    io_.set_mask_path(line_edit::get_text(line_edit_mask));
    on_change();
  });
  connect(line_edit_unit_vector      , &QLineEdit::editingFinished, [&]
  {
    io_.set_unit_vector_path(line_edit::get_text(line_edit_unit_vector));
    on_change();
  });
  connect(button_vervet_defaults     , &QPushButton::clicked      , [&]
  {
    line_edit_file         ->setPlaceholderText("C:/Vervet1818.h5");
    line_edit_transmittance->setPlaceholderText("%Slice%/Microscope/Processed/Registered/NTransmittance");
    line_edit_retardation  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Retardation");
    line_edit_direction    ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Direction");
    line_edit_inclination  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Inclination");
    line_edit_mask         ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Mask");
    line_edit_unit_vector  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/UnitVector");
  });
  connect(button_msa_defaults        , &QPushButton::clicked      , [&]
  {
    line_edit_file         ->setPlaceholderText("C:/MSA0309_s0536-0695.h5");
    line_edit_transmittance->setPlaceholderText("Transmittance");
    line_edit_retardation  ->setPlaceholderText("Retardation");
    line_edit_direction    ->setPlaceholderText("Direction");
    line_edit_inclination  ->setPlaceholderText("Inclination");
    line_edit_mask         ->setPlaceholderText("Mask");
    line_edit_unit_vector  ->setPlaceholderText("UnitVector");
  });
}

void data_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
  logger_->info(std::string("Start successful."));

  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&] (
    const std::array<std::size_t, 3>& offset,
    const std::array<std::size_t, 3>& size  ,
    const std::array<std::size_t, 3>& stride)
  {
    

    on_load();
  });
}
}
