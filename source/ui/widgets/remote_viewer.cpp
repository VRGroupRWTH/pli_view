#include <pli_vis/ui/widgets/remote_viewer.hpp>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cstdint>
#include <filesystem>

#include <pli_vis/third_party/cppzmq/zmq.hpp>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/plugins/local_tractography_plugin.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/primitives/camera.hpp>

#include <image.pb.h>
#include <parameters.pb.h>

namespace pli
{
remote_viewer::remote_viewer(const std::string& address, const std::string& folder, application* owner, interactor* interactor, QWidget* parent) : QLabel(parent), owner_(owner), interactor_(interactor), alive_(true)
{
  setAttribute  (Qt::WidgetAttribute::WA_StaticContents, true);
  setWindowTitle(std::string("Remote Viewer " + address).c_str());
  move          (12 , 32 );
  resize        (640, 480);
  show          ();
  
  address_ = address;
  folder_  = folder ;

  future_ = std::async(std::launch::async, [&]
  {
    auto camera               = owner_->viewer->camera                            ();
    auto data_plugin          = owner_->get_plugin<pli::data_plugin>              ();
    auto color_plugin         = owner_->get_plugin<pli::color_plugin>             ();
    auto tractography_plugin  = owner_->get_plugin<pli::local_tractography_plugin>();

    zmq::context_t context(1);
    zmq::socket_t  socket(context, ZMQ_PAIR);
    socket.connect(std::string(address_));

    while(alive_)
    {
      tt::parameters parameters;

      auto offset = data_plugin->selection_offset();
      auto size   = data_plugin->selection_bounds();
      auto stride = data_plugin->selection_stride();
      if(data_plugin->filepath() != filepath_ || offset  != offset_ || size != size_ || stride != stride_)
      {
        filepath_ = data_plugin->filepath();
        offset_   = offset;
        size_     = size  ;
        stride_   = stride;

        auto data_loading_parameters = parameters.mutable_data_loading();
        data_loading_parameters->set_filepath      (folder_ + std::experimental::filesystem::path(filepath_).filename().string());
        data_loading_parameters->set_dataset_format(filepath_.find("MSA") != std::string::npos ? tt::msa0309 : tt::vervet1818);
        data_loading_parameters->mutable_selection()->mutable_offset()->set_x(offset_[0]);
        data_loading_parameters->mutable_selection()->mutable_offset()->set_y(offset_[1]);
        data_loading_parameters->mutable_selection()->mutable_offset()->set_z(offset_[2]);
        data_loading_parameters->mutable_selection()->mutable_size  ()->set_x(size_  [0]);
        data_loading_parameters->mutable_selection()->mutable_size  ()->set_y(size_  [1]);
        data_loading_parameters->mutable_selection()->mutable_size  ()->set_z(size_  [2]);
        data_loading_parameters->mutable_selection()->mutable_stride()->set_x(stride_[0]);
        data_loading_parameters->mutable_selection()->mutable_stride()->set_y(stride_[1]);
        data_loading_parameters->mutable_selection()->mutable_stride()->set_z(stride_[2]);
      }

      auto seed_offset = tractography_plugin->seed_offset();
      auto seed_size   = tractography_plugin->seed_size  ();
      auto seed_stride = tractography_plugin->seed_stride();
      if (tractography_plugin->step      ()        != step_              || 
          tractography_plugin->iterations()        != iterations_        ||
          tractography_plugin->streamline_radius() != streamline_radius_ ||
          seed_offset                              != seed_offset_       || 
          seed_size                                != seed_size_         || 
          seed_stride                              != seed_stride_       || 
          color_plugin->mode()                     != color_mapping_     ||
          color_plugin->k   ()                     != k_                 )
      {
        step_              = tractography_plugin->step             ();
        iterations_        = tractography_plugin->iterations       ();
        streamline_radius_ = tractography_plugin->streamline_radius();
        seed_offset_       = seed_offset;
        seed_size_         = seed_size  ;
        seed_stride_       = seed_stride;
        color_mapping_     = color_plugin->mode();
        k_                 = color_plugin->k();

        auto particle_tracing_parameters = parameters.mutable_particle_tracing();
        particle_tracing_parameters->set_step      (step_      );
        particle_tracing_parameters->set_iterations(iterations_);
        particle_tracing_parameters->mutable_seeds()->mutable_offset()->set_x(seed_offset_[0]);
        particle_tracing_parameters->mutable_seeds()->mutable_offset()->set_y(seed_offset_[1]);
        particle_tracing_parameters->mutable_seeds()->mutable_offset()->set_z(seed_offset_[2]);
        particle_tracing_parameters->mutable_seeds()->mutable_size  ()->set_x(seed_size_  [0]);
        particle_tracing_parameters->mutable_seeds()->mutable_size  ()->set_y(seed_size_  [1]);
        particle_tracing_parameters->mutable_seeds()->mutable_size  ()->set_z(seed_size_  [2]);
        particle_tracing_parameters->mutable_seeds()->mutable_stride()->set_x(seed_stride_[0]);
        particle_tracing_parameters->mutable_seeds()->mutable_stride()->set_y(seed_stride_[1]);
        particle_tracing_parameters->mutable_seeds()->mutable_stride()->set_z(seed_stride_[2]);
        
        auto color_mapping_parameters = parameters.mutable_color_mapping();
        color_mapping_parameters->set_mapping(tt::color_mapping(color_mapping_));
        color_mapping_parameters->set_k      (k_);

        parameters.mutable_raytracing()->set_streamline_radius(streamline_radius_);
      }
      
      if(camera->translation()          != translation_   ||
         camera->forward    ()          != forward_       ||
         camera->up         ()          != up_            ||
         remote_viewer::size().width () != image_size_[0] ||
         remote_viewer::size().height() != image_size_[1] )
      {
        translation_ = camera->translation();
        forward_     = camera->forward    ();
        up_          = camera->up         ();
        image_size_  = {std::size_t(remote_viewer::size().width()) , std::size_t(remote_viewer::size().height())};

        auto raytracing_parameters = parameters.mutable_raytracing();
        raytracing_parameters->mutable_camera()->mutable_position()->set_x( translation_[0]);
        raytracing_parameters->mutable_camera()->mutable_position()->set_y( translation_[1]);
        raytracing_parameters->mutable_camera()->mutable_position()->set_z( translation_[2]);
        raytracing_parameters->mutable_camera()->mutable_forward ()->set_x(-forward_    [0]);
        raytracing_parameters->mutable_camera()->mutable_forward ()->set_y(-forward_    [1]);
        raytracing_parameters->mutable_camera()->mutable_forward ()->set_z(-forward_    [2]);
        raytracing_parameters->mutable_camera()->mutable_up      ()->set_x( up_         [0]);
        raytracing_parameters->mutable_camera()->mutable_up      ()->set_y( up_         [1]);
        raytracing_parameters->mutable_camera()->mutable_up      ()->set_z( up_         [2]);
        raytracing_parameters->mutable_image_size()->set_x(image_size_[0]);
        raytracing_parameters->mutable_image_size()->set_y(image_size_[1]);
      }

      std::string buffer;
      parameters.SerializeToString(&buffer);

      zmq::message_t request(buffer.size());
      memcpy(request.data(), buffer.data(), buffer.size());
      socket.send(request);

      zmq::message_t reply;
      socket.recv(&reply);

      tt::image image;
      image.ParseFromArray(reply.data(), static_cast<std::int32_t>(reply.size()));

      image_ = QImage(reinterpret_cast<const unsigned char*>(image.data().c_str()), image.size().x(), image.size().y(), QImage::Format_RGBA8888);
      on_render();
    }
  });

  on_render.connect([&] ()
  {
    setPixmap(QPixmap::fromImage(image_));
  });
}
remote_viewer::~remote_viewer()
{
  alive_ = false;
  try
  {
    future_.get();
  }
  catch(...)
  {
    
  }
}

void remote_viewer::closeEvent     (QCloseEvent* event)
{
  QLabel::closeEvent(event);
  on_close();
}
void remote_viewer::keyPressEvent  (QKeyEvent*   event)
{
  interactor_->key_press_handler  (event);
}
void remote_viewer::keyReleaseEvent(QKeyEvent*   event)
{
  interactor_->key_release_handler(event);
}
void remote_viewer::mousePressEvent(QMouseEvent* event)
{
  interactor_->mouse_press_handler(event);
}
void remote_viewer::mouseMoveEvent (QMouseEvent* event)
{
  interactor_->mouse_move_handler (event);
}
}
