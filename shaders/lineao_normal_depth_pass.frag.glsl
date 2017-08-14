#ifndef LINEAO_NORMAL_DEPTH_PASS_FRAG_GLSL_
#define LINEAO_NORMAL_DEPTH_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_normal_depth_pass_frag = R"(\
#version 400

uniform uvec2 screen_size   ;
uniform mat4  model         ;
uniform mat4  view          ;
uniform mat4  projection    ;
in      vec3  vert_direction;
out     vec4  color         ;

vec3 get_world_position()
{
  vec4 normalized_device_coordinates = vec4(
    2.0 * gl_FragCoord.x / screen_size.x - 1.0,
    2.0 * gl_FragCoord.y / screen_size.y - 1.0,
    2.0 * gl_FragCoord.z                 - 1.0,
    1.0);
  vec4 clip_coordinates  = normalized_device_coordinates / gl_FragCoord.w;
  vec4 world_coordinates = inverse(projection * view * model) * clip_coordinates;
  return world_coordinates.xyz;
}

void main()
{
  vec3 T   = normalize(vert_direction);
  vec3 C   = normalize(inverse(view)[3].xyz - get_world_position());
  vec3 TxC = cross(T, C);
  vec3 N   = cross(TxC / length(TxC), T);
  color    = vec4(N, gl_FragCoord.z * gl_FragCoord.w);
}
)";
}

#endif
