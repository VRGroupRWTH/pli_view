#ifndef PLI_VIS_LOGGING_HPP_
#define PLI_VIS_LOGGING_HPP_

#include <third_party/spdlog/spdlog.h>
#include <third_party/spdlog/logger.h>

namespace pli
{
template<typename type>
class loggable
{
public:
  loggable(std::string name = std::string(), std::shared_ptr<spdlog::sinks::sink> sink = nullptr)
  {
    if (name.empty())
      name = typeid(type).name();

    logger_ = spdlog::get(name);
    if (logger_ == nullptr)
      logger_ = sink ? spdlog::create(name, sink) : spdlog::stdout_logger_mt(name);
  }

  void set_sink(std::shared_ptr<spdlog::sinks::sink> sink)
  {
    auto name = logger_->name();
    if (logger_)
      spdlog::drop(name);
    logger_ = sink ? spdlog::create(name, sink) : spdlog::stdout_logger_mt(name);
  }

protected:
  std::shared_ptr<spdlog::logger> logger_;
};
}

#endif
