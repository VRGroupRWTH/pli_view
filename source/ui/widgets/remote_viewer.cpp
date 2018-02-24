#include <pli_vis/ui/widgets/remote_viewer.hpp>

#include <array>
#include <cstdint>

#include <pli_vis/third_party/cppzmq/zmq.hpp>

#include <image.pb.h>
#include <parameters.pb.h>

namespace pli
{
remote_viewer::remote_viewer(QWidget* parent) : QLabel(parent)
{
  setWindowTitle(std::string("Remote Viewer " + address_).c_str());
  show          ();
  setAttribute  (Qt::WA_DeleteOnClose);

  future_ = std::async(std::launch::async, [&]
  {
    zmq::context_t context(1);
    zmq::socket_t  socket(context, ZMQ_PAIR);
    socket.connect(address_);

    while(alive_)
    {
      tt::parameters parameters;
      // TODO: Fill parameters from UI. Either trace or converge.
      
      //// TODO: Set from command line parameters.
      //auto data_loading_parameters = parameters.mutable_data_loading();
      //data_loading_parameters->set_filepath      ("C:/dev/data/pli/Human/MSA0309_s0536-0695.h5");
      //data_loading_parameters->set_dataset_format(tt::msa0309);
      //data_loading_parameters->mutable_selection()->mutable_offset()->set_x(512u);
      //data_loading_parameters->mutable_selection()->mutable_offset()->set_y(512u);
      //data_loading_parameters->mutable_selection()->mutable_offset()->set_z(0u  );
      //data_loading_parameters->mutable_selection()->mutable_size  ()->set_x(128u);
      //data_loading_parameters->mutable_selection()->mutable_size  ()->set_y(128u);
      //data_loading_parameters->mutable_selection()->mutable_size  ()->set_z(128u);
      //data_loading_parameters->mutable_selection()->mutable_stride()->set_x(1u  );
      //data_loading_parameters->mutable_selection()->mutable_stride()->set_y(1u  );
      //data_loading_parameters->mutable_selection()->mutable_stride()->set_z(1u  );
      //
      //// TODO: Set from command line parameters.
      //auto particle_tracking_parameters = parameters.mutable_particle_tracking();
      //particle_tracking_parameters->set_step      (0.5F);
      //particle_tracking_parameters->set_iterations(100u);
      //particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_x(0u  );
      //particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_y(0u  );
      //particle_tracking_parameters->mutable_seeds()->mutable_offset()->set_z(0u  );
      //particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_x(128u);
      //particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_y(128u);
      //particle_tracking_parameters->mutable_seeds()->mutable_size  ()->set_z(128u);
      //particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_x(4u  );
      //particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_y(4u  );
      //particle_tracking_parameters->mutable_seeds()->mutable_stride()->set_z(4u  );
      //
      //// TODO: Set from command line parameters.
      //auto color_mapping_parameters = parameters.mutable_color_mapping();
      //color_mapping_parameters->set_mapping(tt::tkp_hsv);
      //color_mapping_parameters->set_k      (0.5F       );
      //
      //// TODO: Set from command line parameters.
      //auto raytracing_parameters = parameters.mutable_raytracing();
      //raytracing_parameters->mutable_camera()->mutable_position()->set_x( 0.0F  );
      //raytracing_parameters->mutable_camera()->mutable_position()->set_y( 0.0F  );
      //raytracing_parameters->mutable_camera()->mutable_position()->set_z(-100.0F);
      //raytracing_parameters->mutable_camera()->mutable_forward ()->set_x( 0.0F  );
      //raytracing_parameters->mutable_camera()->mutable_forward ()->set_y( 0.0F  );
      //raytracing_parameters->mutable_camera()->mutable_forward ()->set_z( 1.0F  );
      //raytracing_parameters->mutable_camera()->mutable_up      ()->set_x( 0.0F  );
      //raytracing_parameters->mutable_camera()->mutable_up      ()->set_y(-1.0F  );
      //raytracing_parameters->mutable_camera()->mutable_up      ()->set_z( 0.0F  );
      //raytracing_parameters->mutable_image_size()->set_x(1920u);
      //raytracing_parameters->mutable_image_size()->set_y(1080u);

      std::string buffer;
      parameters.SerializeToString(&buffer);

      zmq::message_t request(buffer.size());
      memcpy(request.data(), buffer.data(), buffer.size());
      socket.send(request);

      zmq::message_t reply;
      socket.recv(&reply);

      tt::image image;
      image.ParseFromArray(reply.data(), static_cast<std::int32_t>(reply.size()));

      QImage ui_image;
      ui_image.loadFromData(reinterpret_cast<const unsigned char*>(image.data().c_str()), image.size().x() * image.size().y() * sizeof(std::array<std::uint8_t, 4>));
      setPixmap(QPixmap::fromImage(ui_image));
    }
  });
}
remote_viewer::~remote_viewer()
{
  alive_ = false;
  future_.get();
}
}
