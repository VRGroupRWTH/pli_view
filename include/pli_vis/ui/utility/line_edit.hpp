#ifndef PLI_VIS_LINE_EDIT_HPP_
#define PLI_VIS_LINE_EDIT_HPP_

#include <string>

#include <boost/lexical_cast.hpp>
#include <QLineEdit>

namespace pli
{
class line_edit
{
public:
  template<typename type = std::string>
  static type get_text(QLineEdit* line_edit)
  {
    return boost::lexical_cast<type>(
      !line_edit->text           ().isEmpty    () ?
       line_edit->text           ().toStdString() :
       line_edit->placeholderText().toStdString());
  }
};
}

#endif
