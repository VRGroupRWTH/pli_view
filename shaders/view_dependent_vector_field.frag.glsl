#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_frag = R"(\
#version 450

uniform bool  hsv            = true;
uniform bool  view_dependent = true;
uniform bool  invert         = true;
uniform float rate_of_decay  = 1.0 ;
uniform float cutoff         = 0.25;
uniform mat4  view           ;
out     vec4  frag_color     ;

in vertex_data {
  flat vec3 direction;
} fs_in;
 
vec3 rgb_to_hsv(vec3 color)
{
  vec4  k = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4  p = mix(vec4(color.bg, k.wz), vec4(color.gb, k.xy), step(color.b, color.g));
  vec4  q = mix(vec4(p.xyw, color.r), vec4(color.r, p.yzx), step(p.x, color.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv_to_rgb(vec3 color)
{
  vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(color.xxx + k.xyz) * 6.0 - k.www);
  return color.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), color.y);
}

vec3 to_spherical(vec3 cartesian)
{
  float r = length(cartesian);
  float t = atan  (cartesian.y , cartesian.x);
  float p = acos  (cartesian.z / r);
  return vec3(r, t, p);
}
vec3 to_cartesian(vec3 spherical)
{
  float x = spherical.x * cos(spherical.y) * sin(spherical.z);
  float y = spherical.x * sin(spherical.y) * sin(spherical.z);
  float z = spherical.x * cos(spherical.z);
  return vec3(x, y, z);
}

void main()
{
  vec3 spherical = to_spherical(fs_in.direction);
  if(spherical.y < 0.0)
     spherical.y += radians(180.0);
  if(spherical.y >= radians(180.0))
     spherical.y -= radians(180.0);
  spherical.y = radians(180.0) - spherical.y;

  if(spherical.z < 0.0)
     spherical.z = abs(spherical.z);
  if(spherical.z >= radians( 90.0))
     spherical.z = radians(180.0) - spherical.z;

  float hue        = (spherical.y / radians(180.0));
  float saturation = (spherical.z / radians( 90.0));
  float value      = 1.0;
  
  if(hsv) frag_color = vec4(hsv_to_rgb(vec3(hue, value, saturation)), 1.0);
  else    frag_color = vec4(abs(fs_in.direction.x), abs(fs_in.direction.z), abs(fs_in.direction.y), 1.0);

  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(fs_in.direction)));
    if(invert)
      alpha = 1.0 - alpha;
    if(alpha < cutoff)
      discard;
    frag_color.a = pow(alpha, rate_of_decay);
  }
}
)";
}

#endif
