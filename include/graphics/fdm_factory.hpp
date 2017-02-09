#ifndef PLI_VIS_FDM_POLY_DATA_HPP_
#define PLI_VIS_FDM_POLY_DATA_HPP_

#define _USE_MATH_DEFINES

#include <array>
#include <math.h>

#include <boost/multi_array.hpp>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <convert.hpp>
#include <cush.h>

#include <graphics/color_mappers/rgb.hpp>
#include <utility/base_type.hpp>

namespace pli
{
class fdm_factory
{
public:
  template<typename coefficient_type, typename sample_type = std::array<coefficient_type, 3>>
  static boost::multi_array<sample_type, 4> sample_coefficients(
    const boost::multi_array<coefficient_type, 4>& coefficients     ,
    const std::array<std::size_t, 2>&              sample_dimensions)
  {
    auto shape = coefficients.shape();

    boost::multi_array<sample_type, 4> samples(boost::extents
      [shape[0]]
      [shape[1]]
      [shape[2]]
      [sample_dimensions[0] * sample_dimensions[1]]);

    for (auto x = 0; x < shape[0]; x++)
    {
      for (auto y = 0; y < shape[1]; y++)
      {
        for (auto z = 0; z < shape[2]; z++)
        {
          for (auto lon = 0; lon < sample_dimensions[0]; lon++)
          {
            for (auto lat = 0; lat < sample_dimensions[1]; lat++)
            {
              auto& sample = samples[x][y][z][lon * sample_dimensions[1] + lat];
              sample[1] = 2 * M_PI * coefficient_type(lon) /  sample_dimensions[0];
              sample[2] =     M_PI * coefficient_type(lat) / (sample_dimensions[1] - 1);

              for (auto c = 0; c < coefficients.size(); c++)
                sample[0] += cush::evaluate(c, sample[1], sample[2]) * coefficients[x][y][z][c];

              sample = pli::to_cartesian_coords(sample);
            }
          }
        }
      }
    }
    return samples;
  }

  template<typename points_type, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    boost::multi_array<points_type, 4>& distributions    ,
    const std::array<std::size_t, 2>&   sample_dimensions,
    color_mapper_type                   color_mapper     = color_mapper_type())
  {
    auto poly_data      = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions      = vtkSmartPointer<vtkPoints>           ::New();
    auto colors         = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells          = vtkSmartPointer<vtkCellArray>        ::New();
    auto num_elements   = distributions.num_elements();
    auto num_components = 3;
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements  );
    colors   ->SetNumberOfTuples    (num_elements  );
    cells    ->Allocate             (cells->EstimateSize(num_elements, 4));

    auto index = 0;
    auto shape = distributions.shape();
    for (auto x = 0; x < shape[0]; x++)
    {
      for (auto y = 0; y < shape[1]; y++)
      {
        for (auto z = 0; z < shape[2]; z++)
        {
          auto& distribution = distributions[x][y][z];

          // Normalize samples.
          auto max_distribution = *std::max_element(distribution.begin(), distribution.end(),
          [](const points_type& lhs, const points_type& rhs)
          {
            return sqrt(pow(lhs[0], 2) + pow(lhs[1], 2) + pow(lhs[2], 2)) < 
                   sqrt(pow(rhs[0], 2) + pow(rhs[1], 2) + pow(rhs[2], 2));
          });
          auto max_distribution_length = sqrt(
            pow(max_distribution[0], 2) + 
            pow(max_distribution[1], 2) + 
            pow(max_distribution[2], 2));
          std::transform(distribution.begin(), distribution.end(), distribution.begin(),
          [max_distribution_length] (points_type value)
          {
            value[0] = value[0] / max_distribution_length;
            value[1] = value[1] / max_distribution_length;
            value[2] = value[2] / max_distribution_length;
            return value;
          });

          auto last_index = index;
          for (auto s = 0; s < shape[3]; s++)
          {
            auto position = distribution[s];
            colors->SetTuple(index, (color_mapper.template map<base_type<points_type>::type>(position)).data());
            position[0] += x;
            position[1] += y;
            position[2] += z;
            positions->SetPoint(index, position.data());
            index++;
          }
          for (auto s = 0; s < sample_dimensions[0]; s++)
          {
            for (auto t = 0; t < sample_dimensions[1]; t++)
            {
              vtkIdType indices[4] = 
              {
                last_index +  s                             * sample_dimensions[1] +  t,
                last_index +  s                             * sample_dimensions[1] + (t + 1) % sample_dimensions[1],
                last_index + (s + 1) % sample_dimensions[0] * sample_dimensions[1] + (t + 1) % sample_dimensions[1],
                last_index + (s + 1) % sample_dimensions[0] * sample_dimensions[1] +  t 
              };
              cells->InsertNextCell(4, indices);
            }
          }
        }
      }
    }
    
    poly_data->SetPoints(positions);
    poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
};
}

#endif