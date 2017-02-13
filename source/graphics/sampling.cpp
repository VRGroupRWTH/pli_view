#include /* implements */ <graphics/sampling.hpp>

#define _USE_MATH_DEFINES

#include <math.h>

#include <cush.h>

#include <cuda/convert.h>
#include <cuda/sampler.h>

namespace pli
{
void sample_sphere(
  const std::array <size_t, 2>&            dimensions,
        std::vector<std::array<float, 3>>& points    ,
        std::vector<unsigned>&             indices   )
{
  points .resize(    dimensions[0] * dimensions[1]);
  indices.resize(4 * dimensions[0] * dimensions[1]);

  for (auto lon = 0; lon < dimensions[0]; lon++)
  {
    for (auto lat = 0; lat < dimensions[1]; lat++)
    {
      auto point_index   =      lon + dimensions[0] * lat ;
      auto indices_index = 4 * (lon + dimensions[0] * lat);

      auto& point = points[point_index];
      point[0] = 1.0;
      point[1] = 2 * M_PI * lon /  dimensions[0];
      point[2] =     M_PI * lat / (dimensions[1] - 1);
      point    = to_cartesian_coords(point);
      
      indices[indices_index    ] =  lon                      * dimensions[1] +  lat,
      indices[indices_index + 1] =  lon                      * dimensions[1] + (lat + 1) % dimensions[1],
      indices[indices_index + 2] = (lon + 1) % dimensions[0] * dimensions[1] + (lat + 1) % dimensions[1],
      indices[indices_index + 3] = (lon + 1) % dimensions[0] * dimensions[1] +  lat;
    }
  }
}

void sample_sums(
  const boost::multi_array<float, 4>&                coefficients,
  const std::array<size_t, 2>&                       dimensions  ,
        boost::multi_array<std::array<float, 3>, 4>& points      ,
        boost::multi_array<unsigned, 4>&             indices     )
{
  auto shape = coefficients.shape();

  points.resize(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [dimensions[0] * dimensions[1]]);
  indices.resize(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [4 * dimensions[0] * dimensions[1]]);

  sample(
    { shape[0], shape[1], shape[2] },
    cush::maximum_degree(shape[3]),
    { dimensions[0], dimensions[1] },
              coefficients.data() , 
    (float3*) points      .data() , 
              indices     .data() );
}
}
