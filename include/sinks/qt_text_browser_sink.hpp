#ifndef PLI_VIS_QT_TEXT_BROWSER_SINK_HPP_
#define PLI_VIS_QT_TEXT_BROWSER_SINK_HPP_

#include <QScrollBar>
#include <QTextBrowser>

#include <spdlog/sinks/sink.h>

namespace pli
{
class qt_text_browser_sink : public spdlog::sinks::sink
{
public:
  qt_text_browser_sink (QTextBrowser* text_browser) : text_browser_(text_browser)
  {

  }
  
  void set_text_browser(QTextBrowser* text_browser)
  {
    text_browser_ = text_browser;
  }

  void log  (const spdlog::details::log_msg& message) override
  {
    text_browser_->setText(text_browser_->toPlainText().append(message.formatted.c_str()));
    auto* scroll_bar = text_browser_->verticalScrollBar();
    scroll_bar->setValue(scroll_bar->maximum());
  }
  void flush() override
  {

  }

private:
  QTextBrowser* text_browser_;
};
}

#endif