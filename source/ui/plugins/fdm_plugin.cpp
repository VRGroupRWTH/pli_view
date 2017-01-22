#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#include <limits>

#include <ui/window.hpp>
#include <utility/qt/line_edit_utility.hpp>
#include <utility/spdlog/qt_text_browser_sink.hpp>

namespace pli
{
fdm_plugin::fdm_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
         
  line_edit_block_size_x    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_y    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_z    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_max_order       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_x       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_y       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  //set_sink(std::make_shared<qt_text_browser_sink>(console_)); // FIX This + Viewer update.
  //
  //connect(checkbox_show             , &QCheckBox::stateChanged   , [&](int state)
  //{
  //  show_ = state;
  //  viewer_->update();
  //});
  //connect(line_edit_block_size_x    , &QLineEdit::editingFinished, [&] 
  //{
  //  block_size_[0] = line_edit_utility::get_text<std::size_t>(line_edit_block_size_x);
  //  viewer_->update();
  //});
  //connect(line_edit_block_size_y    , &QLineEdit::editingFinished, [&]
  //{
  //  block_size_[1] = line_edit_utility::get_text<std::size_t>(line_edit_block_size_y);
  //  viewer_->update();
  //});
  //connect(line_edit_block_size_z    , &QLineEdit::editingFinished, [&]
  //{
  //  block_size_[2] = line_edit_utility::get_text<std::size_t>(line_edit_block_size_z);
  //  viewer_->update();
  //});
  //connect(line_edit_histogram_bins_x, &QLineEdit::editingFinished, [&]
  //{
  //  histogram_bins_[0] = line_edit_utility::get_text<std::size_t>(line_edit_histogram_bins_x);
  //  viewer_->update();
  //});
  //connect(line_edit_histogram_bins_y, &QLineEdit::editingFinished, [&]
  //{
  //  histogram_bins_[1] = line_edit_utility::get_text<std::size_t>(line_edit_histogram_bins_y);
  //  viewer_->update();
  //});
  //connect(line_edit_max_order       , &QLineEdit::editingFinished, [&]
  //{
  //  max_order_ = line_edit_utility::get_text<std::size_t>(line_edit_max_order);
  //  viewer_->update();
  //});
  //connect(line_edit_samples_x       , &QLineEdit::editingFinished, [&]
  //{
  //  samples_[0] = line_edit_utility::get_text<std::size_t>(line_edit_samples_x);
  //  viewer_->update();
  //});
  //connect(line_edit_samples_y       , &QLineEdit::editingFinished, [&]
  //{
  //  samples_[1] = line_edit_utility::get_text<std::size_t>(line_edit_samples_y);
  //  viewer_->update();
  //});

  hedgehog_ = vtkSmartPointer<vtkHedgeHog>      ::New();
  mapper_   = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_    = vtkSmartPointer<vtkActor>         ::New();
}

void fdm_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update_viewer();
  });

  owner_->viewer->renderer()->AddActor(actor_);
  owner_->viewer->renderer()->ResetCamera();
}

void fdm_plugin::update_viewer() const
{

}
}
