#ifndef VOLUME_RENDERER_FRAG_GLSL_
#define VOLUME_RENDERER_FRAG_GLSL_

#include <string>

namespace shaders
{
std::string volume_renderer_frag = R"(\
#version 400

const   uint      iterations       = 100u;
uniform vec4      background_color = vec4(0.0, 0.0, 0.0, 1.0);
uniform float     step_size        = 0.01;
uniform uvec2     screen_size      ;
uniform sampler1D transfer_function;
uniform sampler2D exit_points      ;
uniform sampler3D volume           ;
in      vec3      vert_color       ; // Entry point.
out     vec4      frag_color       ;

void main()
{
  vec3 exit_point = texture(exit_points, gl_FragCoord.st / screen_size).xyz;

  if(vert_color == exit_point)
    discard;

  vec3  ray          = (exit_point - vert_color).xyz;
  float ray_length   = length(ray);
  
  vec3  delta        = normalize(ray) * step_size;
  float delta_length = length(delta);

  vec3  current_step = vert_color.xyz;
  float current_length;

  float volume_sample;
  vec4  transfer_function_sample;

  for(uint i = 0; i < iterations; i++)
  {
    volume_sample            = texture(volume, current_step).x;
    transfer_function_sample = texture(transfer_function, volume_sample);

    // Calculate contribution.
    if(transfer_function_sample.a > 0.0)
    {
      transfer_function_sample.a = 1.0 - pow(1.0 - transfer_function_sample.a, step_size);
      frag_color.rgb += (1.0 - frag_color.a) * transfer_function_sample.rgb * transfer_function_sample.a;
      frag_color.a   += (1.0 - frag_color.a) * transfer_function_sample.a;
    }

    current_step   += delta;
    current_length += delta_length;
    
    // Early termination.
    if(current_length >= ray_length)
    {
      frag_color.rgb = frag_color.a * frag_color.rgb + (1.0 - frag_color.a) * background_color.rgb;
      break;
    }
    else if(frag_color.a > 1.0)
    {
      frag_color.a = 1.0;
      break;
    }
  }
}
)";
}


#endif