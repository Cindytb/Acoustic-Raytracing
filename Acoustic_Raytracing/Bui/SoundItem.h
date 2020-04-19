#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <optix.h>
#include "gdt/math/vec.h"
#include "constants.h"

class SoundItem
{
public:
    static int num_mics, num_src, fs, frames_per_buffer;
    static OptixShaderBindingTable sbt;
	static OptixTraversableHandle traversable;
	static OptixPipeline pipeline;
    static int freq_bands, time_bins, num_rays;
    static float hist_bin_size, time_thres, dist_thres, energy_thres, c;
    SoundItem(gdt::vec3f pos, gdt::vec3f orientation);
    void updateSBT(OptixShaderBindingTable sbt);
    SoundItem();

protected:
    gdt::vec3f m_position;
    gdt::vec3f m_orientation;
};