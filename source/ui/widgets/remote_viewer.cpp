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

  future_ = std::async(std::launch::async, [&]
  {
    zmq::context_t context(1);
    zmq::socket_t  socket(context, ZMQ_PAIR);
    socket.connect(address_);

    while(true)
    {
      tt::parameters parameters;
      // TODO: Fill parameters from UI. Either trace or converge.

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
}
