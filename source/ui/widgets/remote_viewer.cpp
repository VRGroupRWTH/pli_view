#include <pli_vis/ui/widgets/remote_viewer.hpp>

#include <array>
#include <cstdint>

#include <pli_vis/third_party/cppzmq/zmq.hpp>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/plugins/local_tractography_plugin.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/primitives/camera.hpp>

#include <image.pb.h>
#include <parameters.pb.h>

namespace pli
{
remote_viewer::remote_viewer(application* owner, QWidget* parent) : QLabel(parent), owner_(owner)
{
  setWindowTitle(std::string("Remote Viewer " + address_).c_str());
  resize        (640, 480);
  show          ();
  
  future_ = std::async(std::launch::async, [&]
  {
    auto camera               = owner_->viewer->camera                            ();
    auto data_plugin          = owner_->get_plugin<pli::data_plugin>              ();
    auto color_plugin         = owner_->get_plugin<pli::color_plugin>             ();
    auto tractography_plugin  = owner_->get_plugin<pli::local_tractography_plugin>();

    zmq::context_t context(1);
    zmq::socket_t  socket(context, ZMQ_PAIR);
    socket.connect(address_);

    while(alive_)
    {
      tt::parameters parameters;

      auto offset = data_plugin->selection_offset();
      auto size   = data_plugin->selection_bounds();
      auto stride = data_plugin->selection_stride();
      auto data_loading_parameters = parameters.mutable_data_loading();
      data_loading_parameters->set_filepath      (data_plugin->filepath());
      data_loading_parameters->set_dataset_format(tt::msa0309);
      data_loading_parameters->mutable_selection()->mutable_offset()->set_x(offset[0]);
      data_loading_parameters->mutable_selection()->mutable_offset()->set_y(offset[1]);
      data_loading_parameters->mutable_selection()->mutable_offset()->set_z(offset[2]);
      data_loading_parameters->mutable_selection()->mutable_size  ()->set_x(size  [0]);
      data_loading_parameters->mutable_selection()->mutable_size  ()->set_y(size  [1]);
      data_loading_parameters->mutable_selection()->mutable_size  ()->set_z(size  [2]);
      data_loading_parameters->mutable_selection()->mutable_stride()->set_x(stride[0]);
      data_loading_parameters->mutable_selection()->mutable_stride()->set_y(stride[1]);
      data_loading_parameters->mutable_selection()->mutable_stride()->set_z(stride[2]);
      
      auto seed_offset = tractography_plugin->seed_offset();
      auto seed_size   = tractography_plugin->seed_size  ();
      auto seed_stride = tractography_plugin->seed_stride();
      auto particle_tracking_parameters = parameters.mutable_particle_tracking();
      particle_tracking_parameters->set_step      (tractography_plugin->step      ());
      particle_tracking_parameters->set_iterations(tractography_plugin->iterations());
      particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_x(seed_offset[0]);
      particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_y(seed_offset[1]);
      particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_z(seed_offset[2]);
      particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_x(seed_size  [0]);
      particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_y(seed_size  [1]);
      particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_z(seed_size  [2]);
      particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_x(seed_stride[0]);
      particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_y(seed_stride[1]);
      particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_z(seed_stride[2]);

      auto color_mapping_parameters = parameters.mutable_color_mapping();
      color_mapping_parameters->set_mapping(tt::color_mapping(color_plugin->mode()));
      color_mapping_parameters->set_k      (color_plugin->k());
      
      auto raytracing_parameters = parameters.mutable_raytracing();
      raytracing_parameters->mutable_camera()->mutable_position()->set_x( camera->translation()[0]);
      raytracing_parameters->mutable_camera()->mutable_position()->set_y( camera->translation()[1]);
      raytracing_parameters->mutable_camera()->mutable_position()->set_z( camera->translation()[2]);
      raytracing_parameters->mutable_camera()->mutable_forward ()->set_x(-camera->forward    ()[0]);
      raytracing_parameters->mutable_camera()->mutable_forward ()->set_y(-camera->forward    ()[1]);
      raytracing_parameters->mutable_camera()->mutable_forward ()->set_z(-camera->forward    ()[2]);
      raytracing_parameters->mutable_camera()->mutable_up      ()->set_x( camera->up         ()[0]);
      raytracing_parameters->mutable_camera()->mutable_up      ()->set_y( camera->up         ()[1]);
      raytracing_parameters->mutable_camera()->mutable_up      ()->set_z( camera->up         ()[2]);
      raytracing_parameters->mutable_image_size()->set_x(remote_viewer::size().width ());
      raytracing_parameters->mutable_image_size()->set_y(remote_viewer::size().height());
      
      std::string buffer;
      parameters.SerializeToString(&buffer);

      zmq::message_t request(buffer.size());
      memcpy(request.data(), buffer.data(), buffer.size());
      socket.send(request);

      zmq::message_t reply;
      socket.recv(&reply);

      tt::image image;
      image.ParseFromArray(reply.data(), static_cast<std::int32_t>(reply.size()));

      QImage ui_image(reinterpret_cast<const unsigned char*>(image.data().c_str()), image.size().x(), image.size().y(), QImage::Format_RGBA8888);
      
      blockSignals(true);
      setPixmap   (QPixmap::fromImage(ui_image));
      blockSignals(false);
    }
  });
}
remote_viewer::~remote_viewer()
{
  alive_ = false;
  future_.get();
}

void remote_viewer::closeEvent(QCloseEvent* event)
{
  QLabel::closeEvent(event);
  on_close();
}
}
