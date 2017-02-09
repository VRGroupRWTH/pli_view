#ifndef PLI_VIS_COLOR_CONVERT_HPP_
#define PLI_VIS_COLOR_CONVERT_HPP_

#include <array>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/toolbox/hsl.hpp>
#include <boost/gil/extension/toolbox/hsv.hpp>

namespace pli
{
template<typename input_type, typename output_type = input_type>
std::array<output_type, 3> rgb_to_hsv(const std::array<input_type, 3>& rgb)
{
  boost::gil::rgb8_pixel_t   rgb8(rgb[0], rgb[1], rgb[2]);
  boost::gil::hsv32f_pixel_t hsv32;
  color_convert(rgb8, hsv32);
  return
  {
    get_color(hsv32, boost::gil::hsv_color_space::hue_t       ()),
    get_color(hsv32, boost::gil::hsv_color_space::saturation_t()),
    get_color(hsv32, boost::gil::hsv_color_space::value_t     ())
  };
}
template<typename input_type, typename output_type = input_type>
std::array<output_type, 3> hsv_to_rgb(const std::array<input_type, 3>& hsv)
{
  boost::gil::hsv32f_pixel_t hsv32(hsv[0], hsv[1], hsv[2]);
  boost::gil::rgb8_pixel_t   rgb8;
  color_convert(hsv32, rgb8);
  return
  {
    get_color(rgb8, boost::gil::red_t  ()),
    get_color(rgb8, boost::gil::green_t()),
    get_color(rgb8, boost::gil::blue_t ())
  };
}

template<typename input_type, typename output_type = input_type>
std::array<output_type, 3> rgb_to_hsl(const std::array<input_type, 3>& rgb)
{
  boost::gil::rgb8_pixel_t   rgb8(rgb[0], rgb[1], rgb[2]);
  boost::gil::hsl32f_pixel_t hsl32;
  color_convert(rgb8, hsl32);
  return
  {
    get_color(hsl32, boost::gil::hsl_color_space::hue_t       ()),
    get_color(hsl32, boost::gil::hsl_color_space::saturation_t()),
    get_color(hsl32, boost::gil::hsl_color_space::lightness_t ())
  };
}
template<typename input_type, typename output_type = input_type>
std::array<output_type, 3> hsl_to_rgb(const std::array<input_type, 3>& hsl)
{
  boost::gil::hsl32f_pixel_t hsl32(hsl[0], hsl[1], hsl[2]);
  boost::gil::rgb8_pixel_t   rgb8;
  color_convert(hsl32, rgb8);
  return
  {
    get_color(rgb8, boost::gil::red_t  ()),
    get_color(rgb8, boost::gil::green_t()),
    get_color(rgb8, boost::gil::blue_t ())
  };
}
}

#endif