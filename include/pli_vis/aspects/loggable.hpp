#ifndef PLI_VIS_LOGGABLE_HPP_
#define PLI_VIS_LOGGABLE_HPP_

#include <spdlog/spdlog.h>

namespace pli
{
template<typename type>
class loggable
{
public:
  explicit loggable(std::string name = std::string(), std::shared_ptr<spdlog::sinks::sink> sink = nullptr)
  {
    if (name.empty())
      name = typeid(type).name();

    logger_ = spdlog::get(name);
    if (!logger_)
      logger_ = sink ? spdlog::create(name, sink) : spdlog::stdout_logger_mt(name);
  }
  virtual ~loggable() = default;

  void set_sink(std::shared_ptr<spdlog::sinks::sink> sink)
  {
    auto name = logger_->name();
    spdlog::drop(name);
    logger_ = sink ? spdlog::create(name, sink) : spdlog::stdout_logger_mt(name);
  }

protected:
  std::shared_ptr<spdlog::logger> logger_;
};
}

#endif
