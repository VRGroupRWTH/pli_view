#ifndef PLI_VIS_TEXT_BROWSER_SINK_HPP_
#define PLI_VIS_TEXT_BROWSER_SINK_HPP_

#include <QScrollBar>
#include <QTextBrowser>
#include <spdlog/sinks/sink.h>

namespace pli
{
class text_browser_sink : public spdlog::sinks::sink
{
public:
  explicit text_browser_sink (QTextBrowser* text_browser) : text_browser_(text_browser)
  {

  }
  
  void log  (const spdlog::details::log_msg& message) override
  {
    text_browser_->append(message.formatted.c_str());
    text_browser_->verticalScrollBar()->setValue(text_browser_->verticalScrollBar()->maximum());
  }
  void flush() override
  {

  }

private:
  QTextBrowser* text_browser_;
};
}

#endif