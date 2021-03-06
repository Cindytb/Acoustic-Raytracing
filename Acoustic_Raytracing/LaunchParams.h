// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
// #include "Bui/Microphone.h"

namespace osc
{
using namespace gdt;

struct TriangleMeshSBTData
{
  vec3f *vertex;
  vec3i *index;
  bool isMic;
  float absorption;
  int micID;
  vec3f pos;
  vec3f orientation;
};

struct LaunchData
{
  bool dummy_launch = false;
  OptixTraversableHandle traversable;
  float* d_transfer_function;
  float *d_histogram;
  float *d_transmitted;
  vec3f pos;
  vec3f orientation;
  int freq_bands, time_bins, buffer_size;
  float dist_thres, hist_bin_size, energy_thres, c, fs;
};

} // namespace osc
