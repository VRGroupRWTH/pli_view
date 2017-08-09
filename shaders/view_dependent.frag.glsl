#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
#version 400

uniform bool  view_dependent = true   ;
uniform bool  invert         = true   ;
uniform float rate_of_decay  = 1.0    ;
uniform float cutoff         = 0.25   ;
uniform float near_plane     = 0.1    ;
uniform float far_plane      = 10000.0;
uniform uvec2 screen_size    ;
uniform mat4  model          ;
uniform mat4  view           ;
uniform mat4  projection     ;
in      vec3  vert_direction ;
out     vec4  frag_color     ;

vec3  get_world_position  ()
{
  vec4 normalized_device_coordinates = vec4(
    2.0 * gl_FragCoord.x / screen_size.x - 1.0,
    2.0 * gl_FragCoord.y / screen_size.y - 1.0,
    2.0 * gl_FragCoord.z                 - 1.0,
    1.0);
  vec4 clip_coordinates  = normalized_device_coordinates / gl_FragCoord.w;
  vec4 world_coordinates = inverse(projection * view) * clip_coordinates;
  return world_coordinates.xyz;
}
vec3  get_line_normal     ()
{
  vec3 T   = normalize(vert_direction);
  vec3 C   = normalize(inverse(view)[3].xyz - get_world_position());
  vec3 TxC = cross(T, C);
  return cross(TxC / length(TxC), T);
}
float get_linearized_depth()
{
  return (2.0 * near_plane) / (near_plane + far_plane - gl_FragCoord.z * (far_plane - near_plane));
}

void main()
{
  frag_color = vec4(abs(vert_direction.x), abs(vert_direction.z), abs(vert_direction.y), 1.0);
  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(vert_direction)));
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
