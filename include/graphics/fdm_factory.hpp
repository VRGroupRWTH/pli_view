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
  template<typename points_type, typename indices_type = unsigned, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const std::array<std::size_t, 2>&   dimensions       ,
    const std::vector<points_type>&     points           ,
    const std::vector<indices_type>&    indices          ,
    color_mapper_type                   color_mapper     = color_mapper_type())
  {
    auto poly_data = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions = vtkSmartPointer<vtkPoints>           ::New();
    auto colors    = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells     = vtkSmartPointer<vtkCellArray>        ::New();
    colors   ->SetNumberOfComponents(3);
    positions->SetNumberOfPoints    (points.size());
    colors   ->SetNumberOfTuples    (points.size());
    cells    ->Allocate             (cells->EstimateSize(points.size(), 4));

    for (auto i = 0; i < points.size(); i++)
    {
      positions->SetPoint(i, points[i].data());
      colors   ->SetTuple(i, (color_mapper.template map<base_type<points_type>::type>(points[i])).data());
    }

    for (auto i = 0; i < indices.size(); i+=4)
    {
      vtkIdType vtk_indices[4] = { indices[i], indices[i + 1], indices[i + 2], indices[i + 3] };
      cells->InsertNextCell(4, vtk_indices);
    }

    poly_data->SetPoints(positions);
    poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
  
  template<typename points_type, typename indices_type = unsigned, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const std::array<std::size_t, 2>&          dimensions   ,
    const boost::multi_array<points_type , 4>& points       ,
    const boost::multi_array<indices_type, 4>& indices      ,
    color_mapper_type                          color_mapper = color_mapper_type())
  {
    auto poly_data      = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions      = vtkSmartPointer<vtkPoints>           ::New();
    auto colors         = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells          = vtkSmartPointer<vtkCellArray>        ::New();
    auto num_elements   = points.num_elements();
    auto num_components = 3;
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements  );
    colors   ->SetNumberOfTuples    (num_elements  );
    cells    ->Allocate             (cells->EstimateSize(num_elements, 4));

    auto index = 0;
    for (auto x = 0; x < points.shape()[0]; x++)
    {
      for (auto y = 0; y < points.shape()[1]; y++)
      {
        for (auto z = 0; z < points.shape()[2]; z++)
        {
          for (auto s = 0; s < points.shape()[3]; s++)
          {
            auto position = points[x][y][z][s];
            colors->SetTuple(index, (color_mapper.template map<base_type<points_type>::type>(position)).data());
            position[0] += 2 * x;
            position[1] += 2 * y;
            position[2] += 2 * z;
            positions->SetPoint(index, position.data());
            index++;
          }
          for (auto i = 0; i < indices.shape()[3]; i += 4)
          {
            vtkIdType vtk_indices[4] = { 
              indices[x][y][z][i    ], 
              indices[x][y][z][i + 1], 
              indices[x][y][z][i + 2], 
              indices[x][y][z][i + 3]};
            cells->InsertNextCell(4, vtk_indices);
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