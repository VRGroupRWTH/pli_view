#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
#version 400

uniform bool      view_dependent = true   ;
uniform bool      invert         = true   ;
uniform float     rate_of_decay  = 1.0    ;
uniform float     cutoff         = 0.25   ;
uniform float     z_near         = 0.1    ;
uniform float     z_far          = 10000.0;
uniform sampler2D depth_texture  ;
uniform uvec2     screen_size    ;
uniform mat4      view           ;
uniform mat4      projection     ;
in      vec3      vert_direction ;
out     vec4      frag_color     ;

vec3  get_world_position  ()
{
  vec4 normalized_device_coordinates = vec4(
    (gl_FragCoord.x / screen_size.x - 0.5) * 2.0,
    (gl_FragCoord.y / screen_size.y - 0.5) * 2.0,
    (gl_FragCoord.z                 - 0.5) * 2.0,
    1.0);
  vec4 clip_coordinates  = normalized_device_coordinates / gl_FragCoord.w;
  vec4 world_coordinates = inverse(projection * view) * clip_coordinates;
  return world_coordinates.xyz;
}
float get_linearized_depth()
{
  float depth = texture(depth_texture, vec2(
    gl_FragCoord.x / screen_size.x, 
    gl_FragCoord.y / screen_size.y)).x;
  return (2.0 * z_near) / (z_far + z_near - depth * (z_far - z_near));
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
