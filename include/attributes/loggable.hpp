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
  loggable()
  {
    auto logger_name = typeid(type).name();
    logger_ = spdlog::get(logger_name);
    if (logger_ == nullptr)
      logger_ = spdlog::stdout_logger_mt(logger_name);
  }
  loggable(std::string name)
  {
    logger_ = spdlog::get(name);
    if (logger_ == nullptr)
      logger_ = spdlog::stdout_logger_mt(name);
  }
protected:
  std::shared_ptr<spdlog::logger> logger_;
};
}

#endif
