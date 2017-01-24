#ifndef PLI_VIS_FDM_POLY_DATA_HPP_
#define PLI_VIS_FDM_POLY_DATA_HPP_

#include <array>

#include <boost/multi_array.hpp>

#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkSphericalTransform.h>
#include <vtkUnsignedCharArray.h>

#include <utility/std/base_type.hpp>
#include <utility/vtk/color_mappers/rgb.hpp>

namespace pli
{
class fdm_factory
{
public:
  template<typename points_type, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const boost::multi_array<points_type, 4>& distributions    ,
    const std::array<std::size_t, 2>&         sample_dimensions,
    color_mapper_type                         color_mapper     = color_mapper_type())
  {
    auto poly_data           = vtkSmartPointer<vtkPolyData>          ::New();
    auto positions           = vtkSmartPointer<vtkPoints>            ::New();
    auto colors              = vtkSmartPointer<vtkUnsignedCharArray> ::New();
    auto cells               = vtkSmartPointer<vtkCellArray>         ::New();
    auto spherical_transform = vtkSmartPointer<vtkSphericalTransform>::New();
    auto num_components      = 3;
    auto num_elements        = distributions.num_elements();
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements);
    colors   ->SetNumberOfTuples    (num_elements);
    cells    ->Allocate             (cells->EstimateSize(num_elements, 4));

    auto index = 0;
    auto shape = distributions.shape();
    for (auto x = 0; x < shape[0]; x++)
    {
      for (auto y = 0; y < shape[1]; y++)
      {
        for (auto z = 0; z < shape[2]; z++)
        {
          auto last_index = index;
          for (auto s = 0; s < shape[3]; s++)
          {
            auto position = distributions[x][y][z][s];
            spherical_transform->TransformPoint(position.data(), position.data());

            colors->SetTuple(index, (color_mapper.template map<base_type<points_type>::type>(position)).data());

            position[0] += x;
            position[1] += y;
            position[2] += z;
            positions->SetPoint(index, position.data());

            index++;
          }
          //for (auto s = 0; s < sample_dimensions[0]; s++)
          //{
          //  for (auto t = 0; t < sample_dimensions[1]; t++)
          //  {
          //    auto latitude  = last_index + s;
          //    auto longitude = last_index + t;
          //
          //    vtkIdType indices[4] = 
          //    {
          //      latitude      * sample_dimensions[1] +  longitude,
          //      latitude      * sample_dimensions[1] + (longitude + 1) % sample_dimensions[1],
          //     (latitude + 1) % sample_dimensions[1] * sample_dimensions[1] + (longitude + 1) % sample_dimensions[1],
          //     (latitude + 1) % sample_dimensions[1] * sample_dimensions[1] +  longitude 
          //    };
          //
          //    cells->InsertNextCell(4, indices);
          //  }
          //}
        }
      }
    }

    poly_data->SetPoints(positions);
    //poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
};
}

#endif