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

#include <graphics/color_mappers/rgb.hpp>
#include <utility/base_type.hpp>

namespace pli
{
class fdm_factory
{
public:
  template<typename points_type, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const std::vector<points_type>&     samples          ,
    const std::array<std::size_t, 2>&   dimensions       ,
    color_mapper_type                   color_mapper     = color_mapper_type())
  {
    auto poly_data = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions = vtkSmartPointer<vtkPoints>           ::New();
    auto colors    = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells     = vtkSmartPointer<vtkCellArray>        ::New();
    colors   ->SetNumberOfComponents(3);
    positions->SetNumberOfPoints    (samples.size());
    colors   ->SetNumberOfTuples    (samples.size());
    cells    ->Allocate             (cells->EstimateSize(samples.size(), 4));

    for (auto i = 0; i < samples.size(); i++)
    {
      positions->SetPoint(i, samples[i].data());
      colors   ->SetTuple(i, (color_mapper.template map<base_type<points_type>::type>(samples[i])).data());
    }

    for (auto s = 0; s < dimensions[0]; s++)
    {
      for (auto t = 0; t < dimensions[1]; t++)
      {
        vtkIdType indices[4] = 
        {
           s                      * dimensions[1] +  t,
           s                      * dimensions[1] + (t + 1) % dimensions[1],
          (s + 1) % dimensions[0] * dimensions[1] + (t + 1) % dimensions[1],
          (s + 1) % dimensions[0] * dimensions[1] +  t 
        };
        cells->InsertNextCell(4, indices);
      }
    }
    
    poly_data->SetPoints(positions);
    poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
  
  template<typename points_type, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    boost::multi_array<points_type, 4>& samples          ,
    const std::array<std::size_t, 2>&   sample_dimensions,
    color_mapper_type                   color_mapper     = color_mapper_type())
  {
    auto poly_data      = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions      = vtkSmartPointer<vtkPoints>           ::New();
    auto colors         = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells          = vtkSmartPointer<vtkCellArray>        ::New();
    auto num_elements   = samples.num_elements();
    auto num_components = 3;
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements  );
    colors   ->SetNumberOfTuples    (num_elements  );
    cells    ->Allocate             (cells->EstimateSize(num_elements, 4));

    auto index = 0;
    auto shape = samples.shape();
    for (auto x = 0; x < shape[0]; x++)
    {
      for (auto y = 0; y < shape[1]; y++)
      {
        for (auto z = 0; z < shape[2]; z++)
        {
          auto& sample = samples[x][y][z];
          
          // Normalize samples.
          auto max_sample = *std::max_element(sample.begin(), sample.end(),
          [](const points_type& lhs, const points_type& rhs)
          {
            return sqrt(pow(lhs[0], 2) + pow(lhs[1], 2) + pow(lhs[2], 2)) < 
                   sqrt(pow(rhs[0], 2) + pow(rhs[1], 2) + pow(rhs[2], 2));
          });
          auto max_sample_length = sqrt(
            pow(max_sample[0], 2) + 
            pow(max_sample[1], 2) + 
            pow(max_sample[2], 2));
          std::transform(sample.begin(), sample.end(), sample.begin(),
          [max_sample_length] (points_type value)
          {
            value[0] = value[0] / max_sample_length;
            value[1] = value[1] / max_sample_length;
            value[2] = value[2] / max_sample_length;
            return value;
          });

          auto last_index = index;
          for (auto s = 0; s < shape[3]; s++)
          {
            auto position = sample[s];
            colors->SetTuple(index, (color_mapper.template map<base_type<points_type>::type>(position)).data());
            position[0] += 2 * x;
            position[1] += 2 * y;
            position[2] += 2 * z;
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