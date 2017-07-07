//-------------------------------------------------------------------------------
// tangent
//
// Copyright(c) 2017 RWTH Aachen University, Germany,
// Virtual Reality & Immersive Visualisation Group.
//-------------------------------------------------------------------------------
//                                 License
//
// This framework is free software : you can redistribute it and / or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// In the future, we may decide to add a commercial license
// at our own discretion without further notice.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.If not, see < http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------
#ifndef DEFAULT_TRACERS_H
#define DEFAULT_TRACERS_H

#include "cartesian_grid.hpp"
#include "basic_trilinear_interpolator.hpp"
#include "runge_kutta_4_integrator.hpp"
#include "trace_recorder.hpp"
#include "dummy_recorder.hpp"
#include "simple_tracer.hpp"
#include "omp_pos_tracer.hpp"

namespace tangent {
  struct TrilinearInterpolationTrait {
    using Data = CartesianGrid;
    using Interpolator = BasicTrilinearInterpolator;
  };

  struct CartesianPointAdvectionTrait {
    using Data = CartesianGrid;
    using Integrator = RungeKutta4Integrator<TrilinearInterpolationTrait>;
    using Recorder = DummyRecorder;
  };

  struct CartesianStreamlineTrait {
    using Data = CartesianGrid;
    using Integrator = RungeKutta4Integrator<TrilinearInterpolationTrait>;
    using Recorder = TraceRecorder;
  };

  using SimpleCartGridPointMappingTracer = SimpleTracer<CartesianPointAdvectionTrait>;
  using SimpleCartGridStreamlineTracer = SimpleTracer<CartesianStreamlineTrait>;

  using OmpCartGridPointMappingTracer = OmpPOSTracer<CartesianPointAdvectionTrait>;
  using OmpCartGridStreamlineTracer = OmpPOSTracer<CartesianStreamlineTrait>;
};

#endif