#ifndef ODF_FIELD_RENDERER_FRAG_GLSL_
#define ODF_FIELD_RENDERER_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string odf_field_renderer_frag = R"(\
#version 400

uniform int   color_mode     = 0    ;
uniform float color_k        = 0.5  ;
uniform bool  color_inverted = false;
in      vec3  vert_direction ;
out     vec4  frag_color     ;

vec3 hue_to_rgb(float hue)
{
  float R = abs(hue * 6 - 3) - 1;
  float G = 2 - abs(hue * 6 - 2);
  float B = 2 - abs(hue * 6 - 4);
  return clamp(vec3(R,G,B), 0, 1);
}
vec3 hsv_to_rgb(vec3 hsv)
{
  vec3 rgb = hue_to_rgb(hsv.x);
  return ((rgb - 1.0) * hsv.y + 1.0) * hsv.z;
}
vec3 hsl_to_rgb(vec3 hsl)
{
  vec3 rgb = hue_to_rgb(hsl.x);
  float C = (1 - abs(2 * hsl.z - 1)) * hsl.y;
  return (rgb - 0.5) * C + hsl.z;
}

vec3 to_spherical(vec3 cartesian)
{
  float r = length(cartesian);
  float t = atan  (cartesian.y , cartesian.x);
  float p = acos  (cartesian.z / r);
  return vec3(r, t, p);
}

vec3 map_color(vec3 direction)
{
  vec3 spherical = to_spherical(direction);

  if(spherical.y <  0.0)            spherical.y += radians(180.0);
  if(spherical.y >= radians(180.0)) spherical.y -= radians(180.0);
  spherical.y = radians(180.0) - spherical.y;

  if(spherical.z < 0.0)             spherical.z = abs(spherical.z);
  if(spherical.z >= radians( 90.0)) spherical.z = radians(180.0) - spherical.z;

  float t = spherical.y / radians(180.0);
  float p = spherical.z / radians(90.0);
  if(color_inverted)
    p = 1.0 - p;

  if(color_mode == 0)
    return hsl_to_rgb(vec3(t, color_k, p));
  if(color_mode == 1)
    return hsl_to_rgb(vec3(t, p, color_k));
  if(color_mode == 2)
    return hsv_to_rgb(vec3(t, color_k, p));
  if(color_mode == 3)
    return hsv_to_rgb(vec3(t, p, color_k));
  return vec3(abs(direction.x), abs(direction.z), abs(direction.y));
}

void main()
{
  frag_color = vec4(map_color(vert_direction), 1.0);
}
)";
}

#endif
