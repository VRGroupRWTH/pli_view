#include <pli_vis/visualization/utility/random_texture.hpp>

#include <algorithm>
#include <random>

namespace pli
{
random_texture::random_texture(const glm::uvec3& size)
{
  bind      ();
  wrap_s    (GL_REPEAT );
  wrap_t    (GL_REPEAT );
  wrap_r    (GL_REPEAT );
  min_filter(GL_NEAREST);
  mag_filter(GL_NEAREST);
  unbind    ();

  generate(size);
}

void random_texture::generate (const glm::uvec3& size)
{
  std::random_device                    random_device;
  std::mt19937                          mersenne_twister(random_device());
  std::uniform_real_distribution<float> distribution;
  
  auto voxel_count = size[0] * size[1] * size[2];
  std::vector<glm::vec4> random_vectors(voxel_count);
  std::generate(random_vectors.begin(), random_vectors.end(), [&mersenne_twister, &distribution]()
  {
    return glm::vec4(
      distribution(mersenne_twister), 
      distribution(mersenne_twister), 
      distribution(mersenne_twister), 
      distribution(mersenne_twister));
  });

  bind     ();
  set_image(GL_RGBA32F, size[0], size[1], size[2], GL_RGBA, GL_FLOAT, random_vectors.data());
  unbind   ();
}
}
